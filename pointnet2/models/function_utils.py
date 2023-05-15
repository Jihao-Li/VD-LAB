import torch
import numpy as np
import torch.nn as nn


def overall_accuracy(cm):
    """
    Compute the overall accuracy.
    :param cm: 
    :return: 
    """
    return np.trace(cm) / cm.sum() * 100.0


def accuracy_per_class(cm):
    """
    Compute the accuracy per class and average
    puts -1 for invalid values (division per 0)
    returns average accuracy, accuracy per class
    :param cm: 
    :return: 
        accuracy_per_class: 
        average_accuracy: 
    """
    # equvalent to for class i to
    # number or true positive of class i (data[target==i]==i).sum()/ number of elements of i (target==i).sum()
    sums = np.sum(cm, axis=1)
    mask = (sums > 0)
    sums[sums == 0] = 1
    accuracy_per_class = np.diag(cm) / sums
    accuracy_per_class[np.logical_not(mask)] = -1
    average_accuracy = accuracy_per_class[mask].mean()
    return average_accuracy * 100.0, accuracy_per_class * 100.0


def iou_per_class(cm, ignore_missing_classes=True):
    """
    Compute the iou per class and average iou
    Puts -1 for invalid values
    returns average iou, iou per class
    :param cm: 
    :param ignore_missing_classes: 
    :return: 
        iou_per_class: 
        average_iou: 
    """
    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    mask = (sums > 0)
    sums[sums == 0] = 1
    iou_per_class = np.diag(cm) / sums
    iou_per_class[np.logical_not(mask)] = -1

    if mask.sum() > 0:
        average_iou = iou_per_class[mask].mean()
    else:
        average_iou = 0

    return average_iou * 100.0, iou_per_class * 100.0


def f1score_per_class(cm):
    """
    Compute f1 scores per class and mean f1.
    puts -1 for invalid classes
    returns average f1 score, f1 score per class
    :param cm: 
    :return: 
        f1score_per_class: 
        average_f1_score: 
    """
    # defined as 2 * recall * prec / (recall + prec)
    sums = (np.sum(cm, axis=1) + np.sum(cm, axis=0))
    mask = (sums > 0)
    sums[sums == 0] = 1
    f1score_per_class = 2 * np.diag(cm) / sums
    f1score_per_class[np.logical_not(mask)] = -1
    average_f1_score = f1score_per_class[mask].mean()
    return average_f1_score * 100.0, f1score_per_class * 100.0


def pfa_per_class(cm):
    """
    Compute the probability of false alarms.
    :param cm: 
    :return: 
        pfa_per_class:
        average_pfa
    """
    sums = np.sum(cm, axis=0)
    mask = (sums > 0)
    sums[sums == 0] = 1
    pfa_per_class = (cm.sum(axis=0) - np.diag(cm)) / sums
    pfa_per_class[np.logical_not(mask)] = -1
    average_pfa = pfa_per_class[mask].mean()
    return average_pfa, pfa_per_class


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


def save_parms(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_model(model, model_path):
    torch.save(model, model_path)


def load_parms(model, model_path):
    model.load_state_dict(torch.load(model_path))


def load_model(model_path):
    return torch.load(model_path)
