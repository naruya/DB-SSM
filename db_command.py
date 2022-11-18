import argparse
import socket
import torch
import numpy as np
import cv2
from tqdm import tqdm
import time
import quaternion
from scipy.spatial.transform import Rotation
from db_ssm.model import SSM
from skvideo.io import vread
from copy import deepcopy
import glob
import os

def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--stamp", type=str, default="Nov16_101241")
    parser.add_argument("--resume_epoch", type=int, default=3000)

    parser.add_argument("--model", type=str, default="rssm")
    parser.add_argument("--s_dim", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--v_dim", type=int, default=4)
    parser.add_argument("--a_dim", type=int, default=0)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument('--min_stddev', type=float, default=1e-5)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--logs", type=str, default="../logs/tonpy-v11")

    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--rsize", type=int, default=256)
    parser.add_argument('--dist', type=float, default=0.8)
    parser.add_argument('--zoom', type=float, default=1.0)
    return parser.parse_args(args)


def f_frame(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = np.zeros(list(img.shape[:2]) + [4])
    out[...,:3] = img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    alpha = np.where(gray < 16, 0, 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
    alpha = cv2.erode(alpha, kernel, iterations=2)
    out[...,3] = alpha
    return out.astype(np.uint8)


def xyzquat2c2w(xyz, quat):
    c2w = np.zeros([4,4])
    c2w[:3,:3] = quaternion.as_rotation_matrix(np.quaternion(*quat))
    c2w[:,3] = np.concatenate([xyz, np.array([1.,])])
    c2w[:3,:3] = c2w[:3,:3].dot(Rotation.from_euler('xyz', (0,0,90), degrees=True).as_matrix())
    return c2w


def main():
    args = get_args()

    model = SSM(args).cuda()
    model.load(args.resume_epoch)

    data_dir = "../data/tonpy-v11/data"
    vid_path = sorted(glob.glob(os.path.join(data_dir, "video", "*.gif")))[0]
    viw_path = sorted(glob.glob(os.path.join(data_dir, "view", "*.npy")))[0]
    x = vread(vid_path).astype(np.uint8)
    v = np.load(viw_path).astype(np.float32)
    a = np.zeros([len(v),1]).astype(np.float32)
    v = np.hstack([v, a])
    x_0 = deepcopy(x[0])
    v_0 = deepcopy(v[0])
    print(x_0.shape, v_0.shape)
    x_t = model.step(v_t=v_0, x_0=x_0)

    flag_first = True
    print("I'm Ready!")

    def connect(s):
        while True:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', args.port))
                s.listen(1)
                conn, addr = s.accept()
                print(addr); break
            except OSError as e:
                print(e); time.sleep(1); continue
        return conn

    # https://docs.python.org/ja/3/library/socket.html
    # https://docs.python.org/ja/3/library/socket.html#example
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        conn = connect(s)

        while True:
            with conn:
                time.sleep(0.03)
                last_data = np.zeros(20)
                img_enc, view_enc = None, None
                while True:
                    try:
                        data = conn.recv(1024)
                    except OSError as e:
                        print(e); time.sleep(1); del conn; conn = connect(s); continue
                    if not data:
                        print("no data"); del conn; conn = connect(s); continue
                    try:
                        decode_data = data.decode("utf-8").split(",")
                        decode_data = [float(s) for s in decode_data]
                    except ValueError as e:
                        print(e); break
                    if not len(decode_data) in [7,8]:
                        print("invalid data"); del conn; conn = connect(s); continue

                    # diff check!
                    diff = np.sum(np.abs(last_data[:len(decode_data)] - decode_data))
                    if diff < 1e-5 and img_enc is not None and view_enc is not None:
                        conn.sendall(view_enc + img_enc)
                        continue

                    last_data[:len(decode_data)] = decode_data
                    xyz = np.array(decode_data[:3]) * args.zoom
                    quat = np.array(decode_data[3:7])
                    mode = int(decode_data[7])
                    dist = np.sum(np.power(xyz, 2)) ** 0.5
                    print(mode, np.round(dist, 2), np.round(xyz, 2), np.round(quat, 2))

                    # c2w = xyzquat2c2w(xyz, quat)
                    ### TODO
                    # c2w[:3,3] = Rotation.from_euler(
                    #     'xyz', (90,-90,0), degrees=True).as_matrix().dot(c2w[:3,3])
                    # c2w[:3,:3] = Rotation.from_euler(
                    #     'xyz', (90,-90,0), degrees=True).as_matrix().dot(c2w[:3,:3])
                    

                    with torch.no_grad():
                        if flag_first:
                            pass

                        xyz = xyz / np.linalg.norm(xyz)
                        v_t = np.append(xyz, 0.).astype(np.float32)
                        img = model.step(v_t=v_t)
                        # img = np.zeros([args.rsize, args.rsize, 3]).astype(np.uint8)
                        img = cv2.resize(img, [args.rsize, args.rsize])
                        # c2w = torch.from_numpy(c2w).float()
                        # img = r.render_persp(c2w.to(device), size, size,
                                             # fx=focal*(dist/args.dist), fast=True)
                    img = f_frame(img)

                    view = np.zeros([6])  # dummy. not used
                    view = ("{} " * 6)[:-1].format(*view).encode()
                    img_enc = cv2.imencode(".png", img)[1].tobytes()
                    view_enc = "{:04}    ".format(len(view)).encode() + view
                    conn.sendall(view_enc + img_enc)


if __name__ == "__main__":
    main()