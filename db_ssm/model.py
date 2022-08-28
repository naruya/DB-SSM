import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from db_ssm.modules import *
from db_ssm.utils import *


class SSM(nn.Module):
    def __init__(self, args):
        super(SSM, self).__init__()

        self.device = device = args.device
        self.s_dim = s_dim = args.s_dim
        self.v_dim = v_dim = args.v_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim
        self.z_dim = z_dim = args.z_dim
        self.args = args

        if self.args.model == "ssm":
            self.prior = nn.DataParallel(
                Prior(s_dim, v_dim, a_dim, args.min_stddev).to(device[0]), device)
            self.posterior = nn.DataParallel(
                Posterior(self.prior, s_dim, h_dim, args.min_stddev).to(device[0]), device)
            self.encoder = nn.DataParallel(
                Encoder(h_dim, args.size).to(device[0]), device)
            self.decoder = nn.DataParallel(
                Decoder(s_dim, args.size).to(device[0]), device)

        elif self.args.model == "rssm":
            self.prior = nn.DataParallel(
                Prior_R(s_dim, z_dim, v_dim, a_dim, args.min_stddev).to(device[0]), device)
            self.posterior = nn.DataParallel(
                Posterior_R(self.prior, s_dim, z_dim, h_dim, args.min_stddev).to(device[0]), device)
            self.encoder = nn.DataParallel(
                Encoder(h_dim, args.size).to(device[0]), device)
            self.decoder = nn.DataParallel(
                Decoder(s_dim, args.size).to(device[0]), device)

        self.distributions = nn.ModuleList([
            self.prior, self.posterior, self.encoder, self.decoder])
        init_weights(self.distributions)
        self.g_optimizer = optim.Adam(self.distributions.parameters())


    def forward(self, x_0, x, v, train=True, return_x=False):
        _B, _T = x.size(0), x.size(1)
        x = x.transpose(0, 1)  # T,B,3,64,64
        v = v.transpose(0, 1)  # T,B,1
        # a = a.transpose(0, 1)  # T,B,1

        # initialize --------------------------------
        xq_hist, xp_hist, q_hist, p_hist, z_hist = [], [], [], [], []

        if self.args.model == "ssm":
            sq_prev = self.init_hidden(x_0)
            s_0 = sq_prev.clone()  # not detach
        elif self.args.model == "rssm":
            sq_prev, z_prev = self.init_hidden(x_0)
            s_0 = sq_prev.clone()  # not detach
            z_0 = z_prev.clone()  # not detach

        # pure forward --------------------------------
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

            q_hist.append(q)
            p_hist.append(p)
            xq_hist.append(xq_t)
            sq_prev = sq_t

            if self.args.model == "rssm":
                z_hist.append(z_t)
                z_prev = z_t

            if return_x:
                sp_t = p.rsample()
                xp_t = self.decoder(sp_t)
                xp_hist.append(xp_t)

        if return_x:
            return torch.stack(xq_hist), torch.stack(xp_hist)

        # calc losses
        g_loss = 0
        s_loss = self.calc_s_loss(q_hist, p_hist)
        x_loss = self.calc_x_loss(xq_hist, x)
        g_loss += s_loss + x_loss
        return_dict = {
            "s_loss": s_loss.item(),
            "x_loss": x_loss.item(),
        }

        if self.args.beta_s_snd is not None:
            s_snd_loss = self.calc_s_snd_loss(q_hist)
            g_loss += s_snd_loss*self.args.beta_s_snd
            return_dict.update({"s_snd_loss": s_snd_loss.item()})

        if self.args.beta_s_over is not None:
            if self.args.model == "ssm":
                s_over_loss = self.calc_s_over_loss(s_0, v, q_hist)
            elif self.args.model == "rssm":
                s_over_loss = self.calc_s_over_loss(s_0, z_0, v, q_hist, z_hist)
            g_loss += s_over_loss*self.args.beta_s_over
            return_dict.update({"s_over_loss": s_over_loss.item()})

        return_dict.update({"loss": g_loss.item()})

        return g_loss, return_dict


    def forward_valid(self, x_0, v):
        _B, _T = v.size(0), v.size(1)
        v = v.transpose(0, 1)  # T,B,1
        # a = a.transpose(0, 1)  # T,B,1

        if self.args.model == "ssm":
            sp_prev = self.init_hidden(x_0)  # not sq_prev
        elif self.args.model == "rssm":
            sp_prev, z_prev = self.init_hidden(x_0)  # not sq_prev

        xv_hist = []
        for t in range(_T):
            v_t = v[t]

            if self.args.model == "ssm":
                p = Normal(*self.prior(sp_prev, v_t))
            elif self.args.model == "rssm":
                p, z_t = Normal_and_Belief(*self.prior(sp_prev, z_prev, v_t))

            # sp_t = p.rsample()
            sp_t = p.mean
            xp_t = self.decoder(sp_t)
            xv_hist.append(xp_t)

            sp_prev = sp_t
            if self.args.model == "rssm":
                z_prev = z_t

        return torch.stack(xv_hist),


    def calc_s_loss(self, q_hist, p_hist):
        loss = 0.
        for q, p in zip(q_hist, p_hist):
            loss += torch.sum(
                kl_divergence(q, p), dim=[1,]).mean()  # p last
        return loss


    def calc_x_loss(self, xq_hist, x):
        loss = 0.
        for xq_t, x_t in zip(xq_hist, x):
            loss += - torch.sum(
                Normal(xq_t, torch.ones(x_t.shape, device=x_t.device)).log_prob(x_t),
                dim=[1,2,3]).mean()
        return loss


    def calc_s_snd_loss(self, q_hist):
        prior_snd = Normal(torch.tensor(0.), torch.tensor(1.))
        loss = 0.
        for q in q_hist:
            loss += torch.sum(
                kl_divergence(prior_snd, q), dim=[1,]).mean()  # q last
        return loss


    def calc_s_over_loss(self, *args):
        """
                        (t=0)          (t=1)          (t=2)        T=3
            (s_0) ->    q[0]     ->    q[1]     ->    q[2]
                   \              \              \
                       p_{0|-1}  ->   p_{1|0}   ->   p_{2|1}
                                  \              \             \
                                      p_{1|-1}  ->   p_{2|0}   (t_init= 1)
                                                 \             \
            |                                        p_{2|-1}  (t_init= 0)
            v                                                  \
          depth                                                (t_init=-1)
                        v[0]           v[1]           v[2]
        """
        loss = 0.

        if self.args.model == "ssm":
            s_0, v, q_hist = args
            _T = len(v)

            for t_init in range(-1, _T-1):
                sp_prev = s_0.detach().clone() if t_init==-1 else q_hist[t_init].mean.detach().clone()

                for depth, t in enumerate(range(t_init+1, _T)):
                    p_i_t = Normal(*self.prior(sp_prev, v[t]))

                    if not depth == 0:
                        q_t = Normal(q_hist[t].loc.detach().clone(), q_hist[t].scale.detach().clone())
                        loss += torch.sum(kl_divergence(q_t, p_i_t), dim=[1,]).mean()

                    sp_prev = p_i_t.rsample()

        elif self.args.model == "rssm":
            s_0, z_0, v, q_hist, z_hist = args
            _T = len(v)

            for t_init in range(-1, _T-1):
                sp_prev = s_0.detach().clone() if t_init==-1 else q_hist[t_init].mean.detach().clone()
                z_prev = z_0.detach().clone() if t_init==-1 else z_hist[t_init].detach().clone()

                for depth, t in enumerate(range(t_init+1, _T)):
                    p_i_t, z_t = Normal_and_Belief(*self.prior(sp_prev, z_prev, v[t]))

                    if not depth == 0:
                        q_t = Normal(q_hist[t].loc.detach().clone(), q_hist[t].scale.detach().clone())
                        loss += torch.sum(kl_divergence(q_t, p_i_t), dim=[1,]).mean()

                    sp_prev = p_i_t.rsample()
                    z_prev = z_t

        return loss


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
        print("save model")

        self = unwrap_module(self)
        save_dir = os.path.join(self.args.logs, self.args.stamp, "weights")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "epoch{:05}.pt".format(epoch))

        save_dict = {}
        save_dict.update({"distributions": self.distributions.state_dict()})
        save_dict.update({"g_optimizer": self.g_optimizer.state_dict()})
        torch.save(save_dict, path)


    def load(self, epoch, load_dir=None):
        print("load model")

        self = unwrap_module(self)
        if not load_dir:
            load_dir = os.path.join(self.args.logs, self.args.stamp, "weights")
        path = os.path.join(load_dir, "epoch{:05}.pt".format(epoch))

        checkpoint = torch.load(path)
        self.distributions.load_state_dict(checkpoint["distributions"])
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
