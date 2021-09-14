import argparse
import subprocess
from datetime import datetime


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stamp", type=str, default=_stamp())
    parser.add_argument("--ghash", type=str, default=_ghash())
    parser.add_argument("--resume_epoch", type=int, default=0)

    parser.add_argument("--model", type=str, default="rssm")
    parser.add_argument("--s_dim", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--v_dim", type=int, default=6)
    parser.add_argument("--a_dim", type=int, default=0)
    parser.add_argument("--h_dim", type=int, default=1024)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument('--beta_s_snd', type=float, default=None)
    parser.add_argument('--beta_s_over', type=float, default=None)
    parser.add_argument('--beta_d_sv', type=float, default=None)
    parser.add_argument('--min_stddev', type=float, default=1e-5)

    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--iters_to_accumulate", type=int, default=1)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--B_val", type=int, default=4)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--T_val", type=int, default=300)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--max_norm', type=float, default=1e+7)
    parser.add_argument('--d_sv_max_norm', type=float, default=1e+7)

    parser.add_argument("--freq_write", type=int, default=10)
    parser.add_argument("--freq_save", type=int, default=10)

    args = parser.parse_args(args)
    return args


def _stamp():
    stamp = datetime.now().strftime("%b%d_%H%M%S")
    return stamp


def _ghash():
    ghash = subprocess.check_output(
        "git rev-parse --short HEAD".split()).strip().decode('utf-8')
    return ghash
