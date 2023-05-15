from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import sys
import time
import torch
import pickle
import shutil
import logging
import argparse
import numpy as np
import torch.nn as nn
import os.path as osp
import data.data_utils as data_utils
import pointnet2.models.function_utils as f_utils

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from pointnet2.models.function_utils import *
from data.isprs_dataloader import ISPRS3DWholeDataset
from pointnet2.models.segmentation_model import SegModel


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


"""mapping of class & label"""
class2label = {
    'Powerline': 0, 'Low_vegetation': 1, 'Impervious_surfaces': 2, 'Car': 3, 'Fence': 4,
    'Roof': 5, 'Facade': 6, 'Shrub': 7, 'Tree': 8
}

label2class = {
    0: 'Powerline', 1: 'Low_vegetation', 2: 'Impervious_surfaces', 3: 'Car', 4: 'Fence',
    5: 'Roof', 6: 'Facade', 7: 'Shrub', 8: 'Tree'
}

# color_map
class2rgb = {
    0: [255, 105, 180],      # Powerline: hot pink
    1: [170, 255, 127],      # Low vegetation: chartreuse
    2: [128, 128, 128],      # Impervious surfaces: gray
    3: [255, 215, 0],        # Car: gold
    4: [0, 191, 255],        # Fence: deep sky blue
    5: [0, 0, 127],          # Roof: blue
    6: [205, 133, 0],        # Facade: orange
    7: [160, 32, 240],       # Shrub: purple
    8: [9, 120, 26],         # Tree: green
}


# parse params
parser = argparse.ArgumentParser(description="Eval")

# data params
parser.add_argument("--num_classes", type=int, default=9)
parser.add_argument("--num_points", type=int, default=4096)
parser.add_argument("--input_channels", type=int, default=5)
parser.add_argument("--use_xyz", type=bool, default=True)
parser.add_argument("--data_root", type=str, default="")
parser.add_argument("--save_idx", type=str, default="1")
parser.add_argument("--num_votes", type=int, default=10)

# model params
parser.add_argument("--radius_list", type=list, default=[2, 4, 8, 16])
parser.add_argument("--npoint_list", type=list, default=[2048, 512, 128, 32])
parser.add_argument("--neighbour_points_list", type=list, default=[32, 32, 32, 32])

# other params
parser.add_argument("--seed", type=int, default=166189)
parser.add_argument("--dropout_rate", type=float, default=0.5)
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--param_path', type=str, default="")
parser.add_argument("--infer_report_freq", type=int, default=20)


def preds2rgb(preds_np):
    """
    
    :param preds_np: np array of preds
    :return: 
    """
    num_points = preds_np.size
    map_rgb_np = np.zeros((num_points, 3))
    for i, ele in enumerate(preds_np):
        map_rgb_np[i] = np.array(class2rgb[ele])

    return map_rgb_np


def model_vote(num_points, inputs, model, vote_preds):
    """
    vote inference
    :param num_points: num points in one batch
    :param inputs: input data
    :param model: seg model
    :param vote_preds: 
    :return: 
    """
    block_num_points = inputs.shape[1]
    num_batches = int(np.ceil(block_num_points / num_points))

    points_size = int(num_batches * num_points)
    replace = False if (points_size - block_num_points <= block_num_points) else True

    point_idxs_repeat = np.random.choice(block_num_points, points_size - block_num_points, replace=replace)
    point_idxs = np.concatenate((range(block_num_points), point_idxs_repeat))
    np.random.shuffle(point_idxs)

    for i in range(num_batches):
        current_idxs = point_idxs[i*num_points:(i+1)*num_points]
        with torch.no_grad():
            logits = model(inputs[:, current_idxs.tolist(), :])
        preds = torch.argmax(logits, dim=2)
        vote_preds[current_idxs, preds] += 1

    return vote_preds


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.save = '{}-{}'.format("./test", args.save_idx)
    args.all_mlp_list = [
        [args.input_channels, 32, 32, 64],
        [64, 64, 64, 128],
        [128, 128, 128, 256],
        [256, 256, 256, 512]
    ]
    args.vdchannels = 128

    # create dir
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    try:
        shutil.copy("./train/eval.py", args.save)
    except:
        print("file route error")

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", args)

    logging.info('Experiment_dir: {}'.format(args.save))
    logging.info("Directory has been created!")

    # load data
    test_merge_block_path = osp.join(args.data_root, "processed_no_rgb", "eval_merge")
    test_transforms = transforms.Compose([data_utils.PointcloudToTensor(), ])
    test_set = ISPRS3DWholeDataset(test_merge_block_path, test_transforms)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)
    logging.info("Test data have been loaded!")

    # load params
    model = SegModel(args.use_xyz, args.input_channels, args.vdchannels, args.radius_list, args.npoint_list,
                     args.neighbour_points_list, args.all_mlp_list, args.num_classes, args.dropout_rate)
    logging.info("param size = %fMB", f_utils.count_parameters_in_MB(model))
    load_parms(model, args.param_path)
    model = model.cuda()

    # # load model
    # model = load_model(args.model_path)
    # logging.info("param size = %fMB", f_utils.count_parameters_in_MB(model))
    # model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    all_points = []

    model.eval()
    objs = AvgrageMeter()
    val_cm = np.zeros((args.num_classes, args.num_classes))
    for step, data in enumerate(test_loader):
        inputs, labels, block_path = data

        inputs = inputs.to("cuda", non_blocking=True)
        labels = labels.to("cuda", non_blocking=True)

        if args.num_votes > 0:
            vote_preds = torch.zeros((inputs.shape[1], args.num_classes)).to("cuda", non_blocking=True)
            for _ in range(args.num_votes):
                vote_preds = model_vote(args.num_points, inputs, model, vote_preds)
            final_preds = vote_preds

        else:
            with torch.no_grad():
                final_preds = model(inputs)
            final_preds = final_preds[0]

        loss = criterion(final_preds.view(labels.numel(), -1), labels.view(-1))

        preds_np = np.argmax(final_preds.cpu().detach().numpy(), axis=1).copy()
        labels_np = labels.cpu().numpy().copy()

        val_cm_ = confusion_matrix(labels_np.ravel(), preds_np.ravel(), labels=list(range(args.num_classes)))
        val_cm += val_cm_

        try:
            inputs_np = np.loadtxt(block_path[0])
        except:
            inputs_np = np.loadtxt(block_path[0], delimiter=",")

        map_rgb_np = preds2rgb(preds_np)
        inputs_preds = np.concatenate((inputs_np[:, 0:3], map_rgb_np), axis=1)
        all_points.append(inputs_preds)

        objs.update(loss.item(), inputs.shape[1])
        if step % args.infer_report_freq == 0:
            logging.info('Infer Step: %04d Loss: %f', step, objs.avg)

    calculate_output_metrics(val_cm, args.num_classes, logging)
    np.savetxt(osp.join(args.save, "preds_map.txt"), np.concatenate(all_points, axis=0))
    logging.info("Save preds map")


def calculate_output_metrics(confusion_matrix, num_classes, logging):
    """
    calculate metrics
    :param confusion_matrix: 
    :param num_classes: num of classes
    :param logging: 
    :return: 
    """
    oa = overall_accuracy(confusion_matrix)
    mf1, f1 = f1score_per_class(confusion_matrix)

    logging.info("**********Total Results**********")
    logging.info("OA: {:9f}, mF1: {:9f}".format(oa, mf1))
    logging.info("Each class Results")

    for i in range(num_classes):
        message = "{:19} Result: F1 {:9f}".format(label2class[i], f1[i])
        logging.info(message)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("Done!")
