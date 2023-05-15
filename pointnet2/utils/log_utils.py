import os
import sys
import glob
import torch
import shutil
import logging
import numpy as np
from pathlib import Path


def create_exp_log_dir(args):
    """

    :param args:
    :return: checkpoints dir and tensorboard dir
    """
    exp_log_dir = Path("./experiments/")
    exp_log_dir.mkdir(exist_ok=True)
    exp_log_dir = exp_log_dir.joinpath(args.task_type)
    exp_log_dir.mkdir(exist_ok=True)
    exp_log_dir = exp_log_dir.joinpath(args.save)
    exp_log_dir.mkdir(exist_ok=True)

    # judge train or eval stage
    train_or_eval = args.save.split("-")[0]
    if train_or_eval == "train":
        checkpoints_dir = exp_log_dir.joinpath("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
        log_txt_name = "train_log.txt"
    else:
        checkpoints_dir = None
        log_txt_name = "eval_log.txt"
    scripts_dir = exp_log_dir.joinpath("scripts")
    scripts_dir.mkdir(exist_ok=True)
    tensorboard_dir = exp_log_dir.joinpath("tensorboard")
    tensorboard_dir.mkdir(exist_ok=True)

    for script in glob.glob('*.py'):
        dst_file = os.path.join(exp_log_dir, "scripts", os.path.basename(script))
        shutil.copyfile(script, dst_file)

    # define log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(os.path.join(exp_log_dir, log_txt_name))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    return str(checkpoints_dir), str(tensorboard_dir)


def save_parms(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_model(model, model_path):
    torch.save(model, model_path)


def load_parms(model, model_path):
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path)["model_state"])


def load_model(model_path):
    return torch.load(model_path)


def accuracy(output, target, topk=(1,)):
    """

    :param output: 
    :param target: 
    :param topk: 
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def count_parameters_in_MB(model):
    """

    :param model: 
    :return: 
    """
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
