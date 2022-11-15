import os
import glob
import random
import numpy as np
from skvideo.io import vread
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_dir, mode, T):
        self.T = T

        vid_paths = sorted(glob.glob(os.path.join(data_dir, "video", "*.gif")))
        viw_paths = sorted(glob.glob(os.path.join(data_dir, "view", "*.npy")))

        if mode == 'train':
            indices = list(np.arange(len(vid_paths)))
            indices = list(set(indices) - set(list(np.arange(0, len(indices), 7))))
        else:
            indices = list(np.arange(0, len(vid_paths), 7))

        vid_paths = np.array(vid_paths)[indices]
        viw_paths = np.array(viw_paths)[indices]

        self.vid = np.array([vread(path) for path in vid_paths]).astype(np.uint8)
        self.viw = np.array([np.load(path) for path in viw_paths]).astype(np.float32)

        print(mode, self.vid.shape, self.viw.shape)

    def __len__(self):
        return len(self.vid)

    def __getitem__(self, idx):
        x = self.vid[idx]
        v = self.viw[idx]
        x = np.transpose(x, [0,3,1,2])
        x_0, x, v = x[0], x[1:], v[1:]
        return x_0, x, v


# validation with "mode" dataset
class MyValidDataset(MyDataset):
    def __init__(self, dataset, T_val):
        self.T = T_val
        self.vid = dataset.vid
        self.viw = dataset.viw


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
