import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_utils import ResnetBlock, actvn


class Prior(nn.Module):
    def __init__(self, s_dim, v_dim, a_dim, min_stddev=0.):
        super(Prior, self).__init__()
        self.min_stddev = min_stddev

        self.fc_loc11 = nn.Linear(s_dim + v_dim + a_dim, s_dim*2)
        self.fc_loc12 = nn.Linear(s_dim*2 + v_dim, s_dim)

        self.fc_scale11 = nn.Linear(s_dim*2, s_dim*2)
        self.fc_scale12 = nn.Linear(s_dim*2, s_dim)

    def forward_shared(self, s_prev, v):
        h = torch.cat([s_prev, v], 1)
        h = F.relu(self.fc_loc11(h))
        loc = self.fc_loc12(torch.cat([h, v], 1))
        scale = self.fc_scale12(F.relu(self.fc_scale11(h)))

        return loc, scale

    def forward(self, s_prev, v):
        loc, scale = self.forward_shared(s_prev, v)
        # TODO: loc = F.tanh(loc)?
        return loc, F.softplus(scale) + self.min_stddev


class Posterior(nn.Module):
    def __init__(self, prior, s_dim, h_dim, min_stddev=0.):
        super(Posterior, self).__init__()
        self.prior = prior
        self.min_stddev = min_stddev
        self.fc1 = nn.Linear(s_dim*2 + h_dim, s_dim*2)
        self.fc21 = nn.Linear(s_dim*2, s_dim)
        self.fc22 = nn.Linear(s_dim*2, s_dim)

    def forward(self, s_prev, v, h):
        if hasattr(self.prior, 'module'):
            prior = self.prior.module
        else:
            prior = self.prior

        loc, scale = prior.forward_shared(s_prev, v)
        h = torch.cat([loc, scale, h], 1)
        h = F.relu(self.fc1(h))
        # TODO: loc = F.tanh(loc)?
        return self.fc21(h), F.softplus(self.fc22(h)) + self.min_stddev


class Encoder(nn.Module):
    def __init__(self, size=256, num_filters=64, max_filters=256):
        super(Encoder, self).__init__()

        s0 = self.s0 = 4
        nf = self.nf = num_filters
        nf_max = self.nf_max = max_filters
        num_layers = int(np.log2(size / s0))

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(num_layers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        nf1 = blocks[-1].conv_1.out_channels

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(3, nf, 3, padding=1)
        self.conv_converter = nn.Conv2d(nf1, nf1, kernel_size=4, padding=0)
        self.fc = nn.Linear(nf1, 1024)

    def forward(self, x):
        out = self.conv_img(x)
        out = self.resnet(out)
        out = self.conv_converter(actvn(out))
        out = out.squeeze(-1).squeeze(-1)
        out = self.fc(actvn(out))
        return out


class Decoder(nn.Module):
    def __init__(self, s_dim, size=256, final_channels=64, max_channels=256):
        super(Decoder, self).__init__()

        s0 = self.s0 = 4
        nf = self.nf = final_channels
        nf_max = self.nf_max = max_channels
        self.s_dim = s_dim
        num_layers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** num_layers)

        self.fc = nn.Linear(s_dim, self.nf0 * s0 * s0)

        blocks = []

        for i in range(num_layers):
            nf0 = min(nf * 2 ** (num_layers - i), nf_max)
            nf1 = min(nf * 2 ** (num_layers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, s):
        batch_size = s.size(0)
        out = self.fc(s)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        return out
