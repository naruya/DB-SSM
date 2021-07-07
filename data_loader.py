import os
import random
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skvideo.io import vread


class MyDataset(Dataset):
    def __init__(self, data_dir, mode, T):
        self.T = T

        _mode = "train" if mode == "train" else "test"
        vid_paths = sorted(glob(os.path.join(data_dir, _mode, "video", "*")))
        viw_paths = sorted(glob(os.path.join(data_dir, _mode, "view", "*")))
        # mot_paths = sorted(glob(os.path.join(data_dir, mode, "motion", "*")))

        self.vid = np.concatenate([vread(path) for path in vid_paths]).astype(np.float32)
        self.viw = np.concatenate([np.load(path) for path in viw_paths]).astype(np.float32)[:, 3:] # xyzrpy
        # self.mot = np.concatenate([np.load(path) for path in mot_paths]).astype(np.float32)
        print(mode, self.vid.shape, self.viw.shape)

        if mode == "train":
            self.viw_mean = self.viw.mean(axis=0)
            self.viw_std = self.viw.std(axis=0)
            self.viw = ((self.viw - self.viw_mean) / self.viw_std)
            np.save(os.path.join(data_dir, "param", "viw_mean.npy"), self.viw_mean)
            np.save(os.path.join(data_dir, "param", "viw_std.npy"), self.viw_std)
            print(self.viw.mean(), self.viw.std())  # check
        else:
            self.viw_mean = np.load(os.path.join(data_dir, "param", "viw_mean.npy"))
            self.viw_std = np.load(os.path.join(data_dir, "param", "viw_std.npy"))
            self.viw = ((self.viw - self.viw_mean) / self.viw_std)
            print(self.viw.mean(), self.viw.std())  # check

        # assert len(self.vid) == len(self.viw) == len(self.mot), \
        assert len(self.vid) == len(self.viw), \
            "incorrect data length detected!!!"

    def __len__(self):
        return len(self.vid) - self.T

    def __getitem__(self, t):
        x = self.vid[t:t+self.T+1]
        x = np.transpose(x, [0,3,1,2])
        x_0, x = x[0], x[1:]
        _viw = self.viw[t+1:t+self.T+1]
        # _mot = self.mot[t+1:t+self.T+1]
        # a = np.concatenate([_viw, _mot], 1)
        a = np.concatenate([_viw,], 1)
        return x_0, x, a


class MyDataLoader(DataLoader):
    def __init__(self, mode, T, args):

        dataset = MyDataset(args.data_dir, mode, T)

        super(MyDataLoader, self).__init__(dataset,
                                           batch_size=args.B,
                                           shuffle=args.shuffle,
                                           drop_last=True,
                                           num_workers=4,
                                           pin_memory=True)
