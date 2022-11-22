import argparse
import socket
import cv2
import torch
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
import os
import glob
import time
from copy import deepcopy
from skvideo.io import vread

from db_ssm.model import SSM


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



# use_sr = False
use_sr = True
if use_sr:
    import sys
    sys.path.append('../db_ssm_sr')
    from sr import SR

    def get_flow(x_p, x_t):
        frame1 = cv2.cvtColor(deepcopy(x_p), cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(deepcopy(x_t), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mask = cv2.bitwise_or(
            cv2.threshold(frame1, 0, 255, cv2.THRESH_BINARY)[1], 
            cv2.threshold(frame2, 0, 255, cv2.THRESH_BINARY)[1],
        )[...,None].astype(np.float32) / 255.
        flow = flow * mask

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        map_x = - mag * np.cos(ang) / 32.
        map_y = - mag * np.sin(ang) / 32.
        flow = np.concatenate([map_x[...,None], map_y[...,None]], axis=2)
        return flow


def main():
    args = get_args()

    model = SSM(args).cuda()
    model.load(args.resume_epoch)

    data_dir = "../data/tonpy-v11/data"
    vid_path = sorted(glob.glob(os.path.join(data_dir, "video", "*.gif")))[2]
    viw_path = sorted(glob.glob(os.path.join(data_dir, "view", "*.npy")))[2]
    x = vread(vid_path).astype(np.uint8)
    v = np.load(viw_path).astype(np.float32)
    a = np.zeros([len(v),1]).astype(np.float32)
    v = np.hstack([v, a])
    x_0 = deepcopy(x[0])
    v_0 = deepcopy(v[0])
    x_t = model.step(v_t=v_0, x_0=x_0)

    if use_sr:
        x_p = deepcopy(x_t)
        xl_p = deepcopy(x_t)
        model_sr = SR()
        model_sr.load(600)

    t = 0
    ano_t = 0.
    print(x_0.shape, v_0.shape, v_0)
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
                        xyz = xyz / np.linalg.norm(xyz)
                        if (t+1) % 10 == 0:
                            ano_t = np.random.uniform(-1,1)
                        v_t = np.append(xyz, ano_t).astype(np.float32)
                        x_t = model.step(v_t=v_t)

                        if use_sr:
                            f_t = get_flow(x_p, x_t)
                            xl_t = model_sr.step(x_p, xl_p, x_t, f_t)
                            xl_t = cv2.resize(xl_t, [args.rsize, args.rsize])
                            img = f_frame(xl_t)
                        else:
                            x_t = cv2.resize(x_t, [args.rsize, args.rsize])
                            img = f_frame(x_t)

                    view = np.zeros([6])  # dummy. not used
                    view = ("{} " * 6)[:-1].format(*view).encode()
                    img_enc = cv2.imencode(".png", img)[1].tobytes()
                    view_enc = "{:04}    ".format(len(view)).encode() + view
                    conn.sendall(view_enc + img_enc)

                    if use_sr:
                        x_p = x_t
                        xl_p = xl_t
                    t += 1


if __name__ == "__main__":
    main()