import os
import logzero
from config import get_args
from utils import *
from model import SSM
from data_looper import MyDataLooper


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    os.makedirs("logzero", exist_ok=True)
    logzero.logfile(os.path.join("logzero", args.timestamp + ".txt"))
    logzero.logger.info("args: " + str(args))

    model = SSM(args)
    train_looper = MyDataLooper(model, args, "train")
    test_looper = MyDataLooper(model, args, "test")

    if args.resume_epoch > 0:
        model.load(args.resume_epoch)

    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        train_looper(epoch)
        test_looper(epoch)

        if epoch % args.freq_save == 0:
            model.save(epoch)
            model.load(epoch)

        if epoch % args.freq_write == 0:
            train_looper.write(epoch)
            test_looper.write(epoch)