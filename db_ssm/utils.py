# --------------------------------

import imageio

def make_gif(frames, filename):
    imageio.mimwrite(filename, frames)


# --------------------------------

def unwrap_module(module):
    if hasattr(module, 'module'):
        return module.module
    else:
        return module


# --------------------------------

from torch import nn

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


# --------------------------------

from torch import nn
from torch.nn import functional as F

# https://github.com/akanimax/Variational_Discriminator_Bottleneck/blob/master/source/vdb/Gan_networks.py
# leaky_relu -> relu
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
        dx = self.conv_0(F.relu(x))
        dx = self.conv_1(F.relu(dx))

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