# --------------------------------

import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# --------------------------------

import moviepy.editor as mpy

def make_gif(frames, filename):
    clip = mpy.ImageSequenceClip(list(frames), fps=30)
    clip.write_gif(filename)