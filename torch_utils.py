import os
import torch
from torch import nn


def save_model(model, epoch):
    if hasattr(model, 'module'):
        model = model.module
    print("save model")
    save_dir = os.path.join("weights", model.args.timestamp)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "epoch{:05}.pt".format(epoch))

    save_dict = {}
    for i, dist in enumerate(model.distributions):
        save_dict.update({"g_net{}".format(i): dist.state_dict()})
    save_dict.update({"g_opt": model.g_optimizer.state_dict()})
    # if model.gan:
    #     save_dict.update({"d_net": model.discriminator.state_dict()})
    #     save_dict.update({"d_opt": model.d_optimizer.state_dict()})
    torch.save(save_dict, path)


def load_model(model, epoch, model_dir=None):
    if hasattr(model, 'module'):
        model = model.module
    print("load model")
    save_dir = os.path.join("weights", model.args.timestamp)
    if model_dir:
        save_dir = os.path.join(model_dir, save_dir)
    path = os.path.join(save_dir, "epoch{:05}.pt".format(epoch))

    checkpoint = torch.load(path)
    for i, dist in enumerate(model.distributions):
        dist.load_state_dict(checkpoint["g_net{}".format(i)])
    model.g_optimizer.load_state_dict(checkpoint["g_opt"])
    # if model.gan:
    #     model.discriminator.load_state_dict(checkpoint["d_net"])
    #     model.d_optimizer.load_state_dict(checkpoint["d_opt"])


# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
def init_weights(model):
    if hasattr(model, 'module'):
        model = model.module

    # print("---- init weights ----")
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.RNN, nn.RNNCell, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            # print("  ", type(m))
            continue
        # print("ok", type(m))


from torch.distributions import Normal


def detach_dist(dist):
    if dist.__class__.__name__ == "Normal":
        loc = dist.loc.detach().clone()
        scale = dist.scale.detach().clone()
        return Normal(loc, scale)
    else:
        raise NotImplementedError


def unwrap_module(module):
    if hasattr(module, 'module'):
        return module.module
    else:
        return module


# https://github.com/akanimax/Variational_Discriminator_Bottleneck/blob/master/source/vdb/Gan_networks.py
from torch import nn
from torch.nn import functional as F


class ResnetBlock(nn.Module):
    """
    Resnet Block Sub-module for the Generator and the Discriminator
    Args:
        :param fin: number of input filters
        :param fout: number of output filters
        :param fhidden: number of filters in the hidden layer
        :param is_bias: whether to use affine conv transforms
    """

    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        """ derived constructor """

        # call to super constructor
        super().__init__()

        # State of the object
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout

        # derive fhidden if not given
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Subsubmodules required by this submodule
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x, alpha=0.1):
        """
        forward pass of the block
        :param x: input tensor
        :param alpha: weight of the straight path
        :return: out => output tensor
        """
        # calculate the shortcut path
        x_s = self._shortcut(x)

        # calculate the straight path
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))

        # combine the two paths via addition
        out = x_s + alpha * dx  # note the use of alpha weighter

        return out

    def _shortcut(self, x):
        """
        helper to calculate the shortcut (residual) computations
        :param x: input tensor
        :return: x_s => output tensor from shortcut path
        """
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    """
    utility helper for leaky Relu activation
    :param x: input tensor
    :return: activation applied tensor
    """
    # out = F.leaky_relu(x, 2e-1)
    out = F.relu(x)
    return out