import os
import glob
import random
import numpy as np
from skvideo.io import vread
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_dir, mode, T):
        self.T = T

        vid_paths = sorted(glob.glob(os.path.join(data_dir, mode, "video", "*")))
        viw_paths = sorted(glob.glob(os.path.join(data_dir, mode, "view", "*")))
        # mot_paths = sorted(glob.glob(os.path.join(data_dir, mode, "motion", "*")))

        self.vid = np.concatenate(
            [vread(path) for path in vid_paths]).astype(np.uint8)
        self.viw = np.concatenate(
            [np.load(path) for path in viw_paths]).astype(np.float32)
        # self.mot = np.concatenate(
        #     [np.load(path) for path in mot_paths]).astype(np.float32)

        # bug of moviepy?
        if np.sum(self.vid[-2] - self.vid[-1]) == 0:
            self.vid = self.vid[:-1]

        self.viw = np.deg2rad(self.viw[:, 3:])  # xyzrpy
        self.viw = np.concatenate([np.sin(self.viw), np.cos(self.viw)], axis=1)

        print(mode, self.vid.shape, self.viw.shape)

        if mode == "train":
            self.viw_mean = 0. # self.viw.mean(axis=0)
            self.viw_std = 1. # self.viw.std(axis=0)
            self.viw = ((self.viw - self.viw_mean) / self.viw_std)
            os.makedirs(os.path.join(data_dir, "param"), exist_ok=True)
            np.save(os.path.join(data_dir, "param", "viw_mean.npy"), self.viw_mean)
            np.save(os.path.join(data_dir, "param", "viw_std.npy"), self.viw_std)
        else:
            self.viw_mean = np.load(os.path.join(data_dir, "param", "viw_mean.npy"))
            self.viw_std = np.load(os.path.join(data_dir, "param", "viw_std.npy"))
            self.viw = ((self.viw - self.viw_mean) / self.viw_std)

        assert len(self.vid) == len(self.viw), \
            "incorrect data length detected!!!"

    def __len__(self):
        return len(self.vid) - self.T

    def __getitem__(self, t):
        x = self.vid[t:t+self.T+1]
        x = np.transpose(x, [0,3,1,2])
        x_0, x = x[0], x[1:]
        v = self.viw[t+1:t+self.T+1]
        # a = self.mot[t+1:t+self.T+1]
        return x_0, x, v


# validation with "mode" dataset
class MyValidDataset(MyDataset):
    def __init__(self, dataset, T):
        self.T = T
        self.vid = dataset.vid
        self.viw = dataset.viw
        # self.mot = dataset.mot


class MyLoader(DataLoader):
    def __init__(self, dataset, B, T):
        if type(dataset) == tuple:
            data_dir, mode = dataset
            dataset = MyDataset(data_dir, mode, T)
            self.mode = mode
        else:
            dataset = MyValidDataset(dataset, T)

        super(MyLoader, self).__init__(dataset,
                                       batch_size=B,
                                       shuffle=True,
                                       drop_last=True,
                                       num_workers=4,
                                       pin_memory=True)
