import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from modules import Prior, Posterior, Prior_R, Posterior_R, Encoder, Decoder, Normal_and_Belief
from torch_utils import init_weights, save_model, load_model, detach_dist
import numpy as np


class SSM(nn.Module):
    def __init__(self, args):
        super(SSM, self).__init__()

        self.device = device = args.device
        self.s_dim = s_dim = args.s_dim
        self.v_dim = v_dim = args.v_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim
        self.z_dim = z_dim = args.z_dim  # only used in RSSM
        self.args = args

        if self.args.model == "ssm":
            self.prior = nn.DataParallel(
                Prior(s_dim, v_dim, a_dim, args.min_stddev).to("cuda"), device)
            self.posterior = nn.DataParallel(
                Posterior(self.prior, s_dim, h_dim, args.min_stddev).to("cuda"), device)
            self.encoder = nn.DataParallel(
                Encoder(h_dim, args.size).to("cuda"), device)
            self.decoder = nn.DataParallel(
                Decoder(s_dim, args.size).to("cuda"), device)

        elif self.args.model == "rssm":
            self.prior = nn.DataParallel(
                Prior_R(s_dim, z_dim, v_dim, a_dim, args.min_stddev).to("cuda"), device)
            self.posterior = nn.DataParallel(
                Posterior_R(self.prior, s_dim, z_dim, h_dim, args.min_stddev).to("cuda"), device)
            self.encoder = nn.DataParallel(
                Encoder(h_dim, args.size).to("cuda"), device)
            self.decoder = nn.DataParallel(
                Decoder(s_dim, args.size).to("cuda"), device)

        self.distributions = nn.ModuleList([
            self.prior, self.posterior, self.encoder, self.decoder])
        init_weights(self.distributions)

        # standard normal distribution (for s_snd_loss)
        self.prior_snd = Normal(torch.tensor(0.), torch.tensor(1.))

        self.g_optimizer = optim.Adam(self.distributions.parameters())


    def forward(self, x_0, x, v, train=True, return_x=False):
        _B, _T = x.size(0), x.size(1)
        x = x.transpose(0, 1)  # T,B,3,64,64
        v = v.transpose(0, 1)  # T,B,1
        # a = a.transpose(0, 1)  # T,B,1

        # Initialize --------------------------------
        xq_hist, xp_hist, q_hist, z_hist = [], [], [], []

        if self.args.model == "ssm":
            sq_prev = self.init_hidden(x_0)
            s_0 = sq_prev.detach().clone()
        elif self.args.model == "rssm":
            sq_prev, z_prev = self.init_hidden(x_0)
            s_0 = sq_prev.detach().clone()
            z_0 = z_prev.detach().clone()

        # SSM Losses --------------------------------
        s_loss, x_loss, s_snd_loss = 0, 0, 0  # step losses

        for t in range(_T):
            x_t, v_t = x[t], v[t]
            h_t = self.encoder(x_t)

            if self.args.model == "ssm":
                p = Normal(*self.prior(sq_prev, v_t))
                q = Normal(*self.posterior(sq_prev, v_t, h_t))
            elif self.args.model == "rssm":
                p, z_t = Normal_and_Belief(*self.prior(sq_prev, z_prev, v_t))
                q, _   = Normal_and_Belief(*self.posterior(sq_prev, z_prev, v_t, h_t))

            sq_t = q.rsample()
            xq_t = self.decoder(sq_t)

            s_loss += torch.sum(
                kl_divergence(q, p), dim=[1,]).mean()  # p last
            x_loss += - torch.sum(
                Normal(xq_t, torch.ones(x_t.shape, device=x_0.device)).log_prob(x_t),
                dim=[1,2,3]).mean()
            s_snd_loss += torch.sum(
                kl_divergence(self.prior_snd, q), dim=[1,]).mean()  # q last

            sq_prev = sq_t
            q_hist.append(detach_dist(q))  # for overshoot

            if self.args.model == "rssm":
                z_prev = z_t
                z_hist.append(z_t.detach().clone())  # for overshoot

            if return_x:
                sp_t = p.rsample()
                xp_t = self.decoder(sp_t)
                xp_hist.append(xp_t)
                xq_hist.append(xq_t)

        if self.args.overshoot:
            if self.args.model == "ssm":
                s_over_loss = self.overshoot(s_0, v, q_hist)
            elif self.args.model == "rssm":
                s_over_loss = self.overshoot(s_0, v, q_hist, z_0, z_hist)
        else:
            s_over_loss = torch.zeros([1])

        # Finalize --------------------------------
        if return_x:
            return torch.stack(xq_hist), torch.stack(xp_hist)

        g_loss, d_loss = 0, 0
        g_loss += s_loss + x_loss + s_over_loss + self.args.beta_snd*s_snd_loss

        return_dict = {
            "loss": g_loss.item(),
            "s_loss": s_loss.item(),
            "x_loss": x_loss.item(),
            "s_snd_loss": s_snd_loss.item(),
            "s_over_loss": s_over_loss.item(),
        }

        return g_loss, d_loss, return_dict


    def forward_valid(self, x_0, v):
        _B, _T = v.size(0), v.size(1)
        v = v.transpose(0, 1)  # T,B,1
        # a = a.transpose(0, 1)  # T,B,1

        if self.args.model == "ssm":
            sq_prev = self.init_hidden(x_0)
        elif self.args.model == "rssm":
            sq_prev, z_prev = self.init_hidden(x_0)

        xv_hist = []
        for t in range(_T):
            v_t = v[t]

            if self.args.model == "ssm":
                p = Normal(*self.prior(sq_prev, v_t))
            elif self.args.model == "rssm":
                p, z_t = Normal_and_Belief(*self.prior(sq_prev, z_prev, v_t))

            # sp_t = p.rsample()
            sp_t = p.mean
            xp_t = self.decoder(sp_t)
            xv_hist.append(xp_t)

            sp_prev = sp_t
            if self.args.model == "rssm":
                z_prev = z_t


        return torch.stack(xv_hist),


    def overshoot(self, s_0, v, _q, z_0=None, _z=None):
        """
                              (t=0)          (t=1)          (t=2)
              q_0 (s_0) -> q_1 (_q[0]) -> q_2 (_q[1]) -> q_3 (_q[2])
                        \              \              \
                             p_{1|0}   ->   p_{2|1}   ->   p_{3|2}
                                       \              \              \
                                            p_{2|0}   ->   p_{3|1}    (i=2)
                                                      \              \
            |                                              p_{3|0}    (i=1)
            v                                                        \
          depth                                                       (i=0)
                              v[0]           v[1]           v[2]
        """

        _T = len(v)
        s_loss = 0.

        for i in range(_T):
            sp_prev = s_0 if i==0 else _q[i-1].mean
            if self.args.model == "rssm":
                z_prev = z_0 if i==0 else _z[i-1]

            for depth, t in enumerate(range(i, _T)):
                if self.args.model == "ssm":
                    p = Normal(*self.prior(sp_prev, v[t]))
                elif self.args.model == "rssm":
                    p, z_t = Normal_and_Belief(*self.prior(sp_prev, z_prev, v[t]))

                if not depth == 0:
                    s_loss += torch.sum(
                        kl_divergence(_q[t], p), dim=[1,]).mean()

                sp_prev = p.rsample()
                if self.args.model == "rssm":
                    z_prev = z_t

        return s_loss


    def init_hidden(self, x_0):
        device = x_0.device

        v_t = torch.zeros(x_0.size(0), self.v_dim).to(device)
        # a_t = torch.zeros(x_0.size(0), self.a_dim).to(device)
        s_prev = torch.zeros(x_0.size(0), self.s_dim).to(device)
        z_prev = torch.zeros(x_0.size(0), self.z_dim).to(device)
        h_t = self.encoder(x_0)

        if self.args.model == "ssm":
            q = Normal(*self.posterior(s_prev, v_t, h_t))
            return q.mean

        elif self.args.model == "rssm":
            q, z_t = Normal_and_Belief(*self.posterior(s_prev, z_prev, v_t, h_t))
            return q.mean, z_t


    @torch.no_grad()
    def sample_x(self, x_0, x, v, valid=False):
        x_list = []                    # numpy
        _x_list = [x.transpose(0, 1)]  # torch

        if not valid:
            _x_list += self.forward(x_0, x, v, False, return_x=True)
        else:
            _x_list += self.forward_valid(x_0, v)

        for _x in _x_list:
            _x = torch.clamp(_x, 0, 1)
            _x = _x.transpose(0, 1).detach().cpu().numpy()  # BxT
            _x = (np.transpose(_x, [0,1,3,4,2]) * 255).astype(np.uint8)
            x_list.append(_x)

        return x_list


    # for simulation
    @torch.no_grad()
    def step(self, v_t=None, a_t=None, x_0=None):
        if x_0 is not None:
            x_0 = np.transpose([x_0], [0,3,1,2])
            x_0 = torch.from_numpy(x_0).to(self.device[0]).float() / 255.
            if self.args.model == "ssm":
                s_t = self.init_hidden(x_0)
            elif self.args.model == "rssm":
                s_t, z_t = self.init_hidden(x_0)
        else:
            v_t = v_t[np.newaxis]
            v_t = torch.from_numpy(v_t).to(self.device[0])

            if self.args.model == "ssm":
                p = Normal(*self.prior(self.s_t, v_t))
            elif self.args.model == "rssm":
                p, z_t = Normal_and_Belief(*self.prior(self.s_t, self.z_t, v_t))
            s_t = p.mean  # use mean

        x_t = self.decoder(s_t)
        x_t = torch.clamp(x_t, 0, 1)
        x_t = x_t.cpu().detach().numpy()
        x_t = np.transpose(x_t, [0,2,3,1])[0]
        x_t = (x_t * 255.).astype(np.uint8)

        self.s_t = s_t
        if self.args.model == "rssm":
            self.z_t = z_t
        return x_t


    def save(self, epoch):
        save_model(self, epoch)


    def load(self, epoch, model_dir=None):
        load_model(self, epoch, model_dir)
