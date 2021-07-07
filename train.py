import os
import logzero
import torch
from config import get_args
from utils import set_seed
from model import SSM
from data_looper import MyDataLooper
from torch_utils import save_model, load_model


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    os.makedirs("logzero", exist_ok=True)
    logzero.loglevel(20)
    logzero.logfile(os.path.join("logzero", args.timestamp + ".txt"), loglevel=20)
    logzero.logger.info("args: " + str(args))

    model = SSM(args)
    train_looper = MyDataLooper(model, args.T, args, "train")
    test_looper = MyDataLooper(model, args.T, args, "test")
    valid_looper = MyDataLooper(model, args.T_val, args, "valid")

    if args.load_epoch:
        resume_epoch = args.load_epoch + 1
        load_model(model, args.load_epoch)
    else:
        resume_epoch = 1

    for epoch in range(resume_epoch, args.epochs + 1):
        train_looper(epoch)
        test_looper(epoch)
        valid_looper(epoch)

        if epoch % args.freq_output == 0:
            train_looper.output(epoch)
            test_looper.output(epoch)
            valid_looper.output(epoch)

        if epoch % args.freq_save == 0:
            save_model(model, epoch)
            load_model(model, epoch)
