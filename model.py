import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from modules import Prior, Posterior, Encoder, Decoder
from torch_utils import init_weights, save_model, load_model
import numpy as np


class SSM(nn.Module):
    def __init__(self, args):
        super(SSM, self).__init__()

        self.s_dim = s_dim = args.s_dim
        self.v_dim = v_dim = args.v_dim
        self.a_dim = a_dim = args.a_dim
        self.h_dim = h_dim = args.h_dim
        self.device = args.device
        self.args = args

        self.prior = torch.nn.DataParallel(
            Prior(s_dim, v_dim, a_dim, args.min_stddev).to(self.device[0]),
            self.device)
        self.posterior = torch.nn.DataParallel(
            Posterior(self.prior, s_dim, h_dim, args.min_stddev).to(self.device[0]),
            self.device)
        self.encoder = torch.nn.DataParallel(
            Encoder(args.size).to(self.device[0]),
            self.device)
        self.decoder = torch.nn.DataParallel(
            Decoder(s_dim, args.size).to(self.device[0]),
            self.device)

        self.distributions = nn.ModuleList([
            self.prior, self.posterior, self.encoder, self.decoder])
        init_weights(self.distributions)

        # for s_aux_loss
        self.prior01 = Normal(torch.tensor(0.), scale=torch.tensor(1.))

        self.g_optimizer = optim.Adam(self.distributions.parameters())


    def forward(self, x_0, x, v, train=True, return_x=False):
        _B, _T = x.size(0), x.size(1)
        x = x.transpose(0, 1)  # T,B,3,64,64
        v = v.transpose(0, 1)  # T,B,1
        # a = a.transpose(0, 1)  # T,B,1
        sq_prev = self.sample_s_0(x_0)

        s_loss, x_loss, s_aux_loss = 0, 0, 0

        _xq, _xp = [], []
        for t in range(_T):
            x_t, v_t = x[t], v[t]
            h_t = self.encoder(x_t)

            q = Normal(*self.posterior(sq_prev, v_t, h_t))
            p = Normal(*self.prior(sq_prev, v_t))
            sq_t = q.rsample()
            xq_t = self.decoder(sq_t)

            # SSM Losses
            s_loss += torch.sum(
                kl_divergence(q, p), dim=[1,]).mean()
            x_loss += - torch.sum(
                Normal(xq_t, torch.ones(x_t.shape, device=x_0.device)).log_prob(x_t),
                dim=[1,2,3]).mean()
            s_aux_loss += kl_divergence(
                q, self.prior01).mean()

            sq_prev = sq_t

            if return_x:
                sp_t = p.rsample()
                xp_t = self.decoder(sp_t)
                _xp.append(xp_t)
                _xq.append(xq_t)

        if return_x:
            return torch.stack(_xq), torch.stack(_xp)

        g_loss, d_loss = 0., 0.
        g_loss += s_loss + x_loss

        return_dict = {
            "loss": g_loss.item(),
            "s_loss": s_loss.item(),
            "x_loss": x_loss.item(),
            "s_aux_loss": s_aux_loss.item(),
        }
        return g_loss, d_loss, return_dict


    def forward_valid(self, x_0, v):
        _B, _T = v.size(0), v.size(1)
        v = v.transpose(0, 1)  # T,B,1
        # a = a.transpose(0, 1)  # T,B,1
        sp_prev = self.sample_s_0(x_0)

        _xv = []
        for t in range(_T):
            v_t = v[t]
            p = Normal(*self.prior(sp_prev, v_t))
            sp_t = p.rsample()
            xp_t = self.decoder(sp_t)
            sp_prev = sp_t
            _xv.append(xp_t)

        return torch.stack(_xv),


    def sample_s_0(self, x_0):
        device = x_0.device

        # dummy
        v_t = torch.zeros(x_0.size(0), self.v_dim).to(device)
        # a_t = torch.zeros(x_0.size(0), self.a_dim).to(device)
        s_prev = torch.zeros(x_0.size(0), self.s_dim).to(device)
        h_t = self.encoder(x_0)
        s_t = Normal(*self.posterior(s_prev, v_t, h_t)).mean
        return s_t


    def sample_x(self, x_0, x, v, valid=False):
        with torch.no_grad():
            x_list = []  # numpy
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
            s_t = self.sample_s_0(x_0)
        else:
            v_t = v_t[np.newaxis]
            v_t = torch.from_numpy(v_t).to(self.device[0])
            # if not self.args.no_motion:
            #     a_t = a_t[np.newaxis]
            #     a_t = torch.from_numpy(a_t).to(self.device[0])
            s_t = self.prior(self.s_t, v_t)[0]  # use mean

        x_t = self.decoder(s_t)
        x_t = torch.clamp(x_t, 0, 1)
        x_t = x_t.cpu().detach().numpy()
        x_t = np.transpose(x_t, [0,2,3,1])[0]
        x_t = (x_t * 255.).astype(np.uint8)

        self.s_t = s_t
        return x_t


    def save(self, epoch):
        save_model(self, epoch)


    def load(self, epoch, model_dir=None):
        load_model(self, epoch, model_dir)
