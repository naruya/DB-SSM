import os
import logzero
import numpy
import torch
from db_ssm.model import SSM
from db_ssm.datasets import MyLoader
from db_ssm.trainer import MyLooper
from db_ssm.config import get_args


if __name__ == "__main__":
    args = get_args()

    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(os.path.join(args.logs, args.stamp), exist_ok=True)
    logzero.logfile(os.path.join(args.logs, args.stamp, "log.txt"))
    logzero.logger.info(args)

    model = SSM(args)

    train_loader = MyLoader((args.data, "train"), args.B, args.T)
    test_loader = MyLoader((args.data, "test"), args.B, args.T)
    train_valid_loader = MyLoader(train_loader.dataset, args.B_val, args.T_val)
    test_valid_loader = MyLoader(test_loader.dataset, args.B_val, args.T_val)

    train_looper = MyLooper(model, train_loader, train_valid_loader, args)
    test_looper = MyLooper(model, test_loader, test_valid_loader, args)

    if args.resume_epoch > 0:
        model.load(args.resume_epoch)

    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        train_looper(epoch)
        test_looper(epoch)

        if epoch % args.freq_valid == 0:
            train_looper.valid(epoch)
            test_looper.valid(epoch)

        if epoch % args.freq_save == 0:
            model.save(epoch)
            model.load(epoch)