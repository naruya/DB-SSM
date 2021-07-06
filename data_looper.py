import os
from utils import *
from logzero import logger
import numpy as np
import torch
from data_loader import MyDataLoader


class MyDataLooper(object):
    def __init__(self, model, args, mode):
        self.mode = mode
        self.device = args.device
        self.iters_to_accumulate = args.iters_to_accumulate

        self.loader = MyDataLoader(mode, args)

        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model


    def __call__(self, epoch):
        self.i, N, summ = 0, 0, None

        for x_0, x, a in self.loader:
            x_0 = x_0.to(self.device[0]).float() / 255.
            x = x.to(self.device[0]).float() / 255.
            a = a.to(self.device[0])

            if self.mode == "train":
                return_dict = self._train(x_0, x, a)
            else:
                return_dict = self._test(x_0, x, a)

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

        # evaluation
        _x, _xq, _xp = self.model.sample_x(x_0[:4], x[:4], a[:4])

        path = "output/{}/epoch{:05}/".format(self.model.args.timestamp, epoch)
        os.makedirs(path, exist_ok=True)
        for i in range(4):
            make_gif(_x[i], path + "{}_true{:02}.gif".format(self.mode, i))
            make_gif(_xq[i], path + "{}_pred-q{:02}.gif".format(self.mode, i))
            make_gif(_xp[i], path + "{}_pred-p{:02}.gif".format(self.mode, i))


    def _train(self, x_0, x, a):
        model = self.model
        model.train()

        model.g_optimizer.zero_grad()
        if model.gan:
            model.d_optimizer.zero_grad()

        g_loss, d_loss, return_dict = model.forward(x_0, x, a, True)
        g_loss = g_loss / self.iters_to_accumulate
        g_loss.backward()
        if model.gan:
            d_loss = d_loss / self.iters_to_accumulate
            d_loss.backward()

        if (self.i + 1) % self.iters_to_accumulate == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.distributions.parameters(), 1e+6)
            return_dict.update({"g_grad_norm": grad_norm.item()})
            model.g_optimizer.step()
            if model.gan:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.discriminator.parameters(), 1e+3)
                return_dict.update({"d_grad_norm": grad_norm.item()})
                model.d_optimizer.step()

            logger.info("({}) Iter: {}/{} {}".format(
                self.mode, self.i+1, len(self.loader), return_dict))

        return return_dict


    def _test(self, x_0, x, a):
        model = self.model
        model.eval()

        with torch.no_grad():
            g_loss, d_loss, return_dict = model.forward(x_0, x, a, False)

        return return_dict
