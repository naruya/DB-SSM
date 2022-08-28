import os
import glob
import random
import numpy as np
from skvideo.io import vread
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_dir, mode, T):
        self.T = T
        self.H = 1

        vid_paths = sorted(glob.glob(os.path.join(data_dir, mode, "video", "*.gif")))
        viw_paths = sorted(glob.glob(os.path.join(data_dir, mode, "view", "*.npy")))
        # mot_paths = sorted(glob.glob(os.path.join(data_dir, mode, "motion", "*.npy")))

        # use_head
        # self.vid = np.array([vread(path) for path in vid_paths]).astype(np.uint8)
        self.vid = np.array([vread(path)[:self.T+self.H] for path in vid_paths]).astype(np.uint8)
        self.viw = np.array([np.load(path) for path in viw_paths]).astype(np.float32)
        # self.mot = np.array([np.load(path) for path in mot_paths])]).astype(np.float32)

        # bug of moviepy?
        # if np.sum(self.vid[-2] - self.vid[-1]) == 0:
        #     self.vid = self.vid[:-1]

        self.viw = np.deg2rad(self.viw[..., 3:])  # xyzrpy
        self.viw = np.concatenate([np.sin(self.viw), np.cos(self.viw)], axis=2)

        assert len(self.vid) == len(self.viw), \
            "incorrect data length detected!!!"
        self.N, self.L = self.vid.shape[:2]
        self.N_per_vid = self.L - self.T + 1 - self.H

        print(mode, self.vid.shape, self.viw.shape)

    def __len__(self):
        return self.N * self.N_per_vid

    def __getitem__(self, idx_t):
        idx = idx_t // self.N_per_vid
        t = idx_t % self.N_per_vid
        x = self.vid[idx,t:t+self.T+1]
        x = np.transpose(x, [0,3,1,2])
        x_0, x = x[0], x[1:]
        v = self.viw[idx,t+1:t+self.T+1]
        # a = self.mot[t+1:t+self.T+1]
        return x_0, x, v


# validation with "mode" dataset
class MyValidDataset(MyDataset):
    def __init__(self, dataset, T_val):
        self.T = T_val
        self.N = dataset.N
        self.N_per_vid = dataset.N_per_vid
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
