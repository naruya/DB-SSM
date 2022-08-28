import os
from logzero import logger
import numpy as np
import torch
from db_ssm.utils import *


class MyLooper(object):
    def __init__(self, model, loader, valid_loader, args):
        self.mode = loader.mode
        self.device = args.device
        self.iters_to_accumulate = args.iters_to_accumulate

        self.model = unwrap_module(model)
        self.loader = loader
        self.valid_loader = valid_loader
        self.args = args

        if self.mode == "train":
            self.step = self.train_step
        elif self.mode == "test":
            self.step = self.test_step


    def _toTensor(self, x_0, x, v):
        x_0 = x_0.to(self.device[0]).float() / 255.
        x = x.to(self.device[0]).float() / 255.
        v = v.to(self.device[0])
        return x_0, x, v


    def __call__(self, epoch):
        self.i, N, summ = 0, 0, None

        for x_0, x, v in self.loader:
            x_0, x, v = self._toTensor(x_0, x, v)
            return_dict = self.step(x_0, x, v, epoch)

            if summ is None:
                keys = return_dict.keys()
                summ = dict(zip(keys, [0] * len(keys)))

            # update summary
            for k in summ.keys():
                v = return_dict[k]
                summ[k] += v * x.size(0)

            self.i += 1
            N += x.size(0)

        # write summary
        for k, v in summ.items():
            summ[k] = v / N
        logger.info("({}) Epoch: {} {}".format(self.mode, epoch, summ))


    def train_step(self, x_0, x, v, epoch):
        model = self.model
        model.train()

        # zero grad
        model.g_optimizer.zero_grad()

        # backward
        g_loss, return_dict = model.forward(x_0, x, v, True)
        g_loss = g_loss / self.iters_to_accumulate
        g_loss.backward()  # add grads (no param update here)

        # step
        if (self.i + 1) % self.iters_to_accumulate == 0:
            max_norm = self.args.max_norm
            # max_norm = self.args.max_norm if epoch > 50 else 1e+7
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.distributions.parameters(), max_norm)
            return_dict.update({"g_grad_norm": grad_norm.item()})
            model.g_optimizer.step()  # update params

            logger.info("({}) Iter: {}/{} {}".format(
                self.mode, self.i+1, len(self.loader), return_dict))

        return return_dict


    def test_step(self, x_0, x, v, epoch):
        model = self.model
        model.eval()

        with torch.no_grad():
            g_loss, d_loss, return_dict = model.forward(x_0, x, v, False)

        if (self.i + 1) % self.iters_to_accumulate == 0:
            logger.info("({}) Iter: {}/{} {}".format(
                self.mode, self.i+1, len(self.loader), return_dict))

        return return_dict


    def valid(self, epoch):
        model = self.model
        model.eval()

        path = os.path.join(self.args.logs, self.args.stamp, "outputs", "epoch{:05}".format(epoch))
        os.makedirs(path, exist_ok=True)

        try:
            x_0, x, v = self._toTensor(*iter(self.loader).next())
            M = min(4, len(x))
            _x, _xq, _xp = model.sample_x(x_0[:M], x[:M], v[:M])

            for i in range(M):
                make_gif(_x[i], os.path.join(path, "{}_true{:02}.gif".format(self.mode, i)))
                make_gif(_xq[i], os.path.join(path, "{}_qsample{:02}.gif".format(self.mode, i)))
                make_gif(_xp[i], os.path.join(path, "{}_psample{:02}.gif".format(self.mode, i)))

            x_0, x, v = self._toTensor(*iter(self.valid_loader).next())
            M = min(4, len(x))
            _x, _xv = model.sample_x(x_0[:M], x[:M], v[:M], valid=True)

            for i in range(M):
                make_gif(_x[i], os.path.join(path, "{}-valid_true{:02}.gif".format(self.mode, i)))
                make_gif(_xv[i], os.path.join(path, "{}-valid_pred{:02}.gif".format(self.mode, i)))

        except ValueError as e:
            logger.warning(e)