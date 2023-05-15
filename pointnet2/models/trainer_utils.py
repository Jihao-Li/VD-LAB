import tqdm
import torch
import shutil
import numpy as np
from pointnet2.models.function_utils import *


def save_parms(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_model(model, model_path):
    torch.save(model, model_path)


def load_parms(model, model_path):
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path)["model_state"])


def load_model(model_path):
    return torch.load(model_path)


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    state = {
        'epoch': epoch, 'it': it, 'best_prec': best_prec,
        'model_state': model_state, 'optimizer_state': optim_state
    }

    return state, model


def save_checkpoint(state, model, filename='checkpoint'):
    """

    :param state: 
    :param model: 
    :param filename: 
    :return: 
    """
    model_name = "{}_model.pt".format(filename)
    torch.save(model, model_name)

    weights_name = "{}_weights.pth.tar".format(filename)
    torch.save(state, weights_name)


class Trainer(object):
    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    eval_frequency : int
        How often to run an eval
    log_name : str
        Name of file to output tensorboard_logger to
    """
    def __init__(self, model, model_fn, optimizer, checkpoint_name="ckpt", best_name="best",
                 lr_scheduler=None, bnm_scheduler=None, eval_frequency=-1, viz=None,
                 logging=None, num_classes=None):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = \
            (model, model_fn, optimizer, lr_scheduler, bnm_scheduler)

        self.checkpoint_name, self.best_name = checkpoint_name, best_name
        self.eval_frequency = eval_frequency

        self.training_best, self.eval_best = {}, {}
        self.viz = viz
        self.logging = logging
        self.num_classes = num_classes

    @staticmethod
    def _decode_value(v):
        if isinstance(v[0], float):
            return np.mean(v)
        elif isinstance(v[0], tuple):
            if len(v[0]) == 3:
                num = [l[0] for l in v]
                denom = [l[1] for l in v]
                w = v[0][2]
            else:
                num = [l[0] for l in v]
                denom = [l[1] for l in v]
                w = None

            return np.average(
                np.sum(num, axis=0) / (np.sum(denom, axis=0) + 1e-6), weights=w)
        else:
            raise AssertionError("Unknown type: {}".format(type(v)))

    def _train_it(self, it, batch):
        self.model.train()

        if self.bnm_scheduler is not None:
            self.bnm_scheduler.step(it)

        self.optimizer.zero_grad()
        _, loss, res, train_cm_ = self.model_fn(self.model, batch)

        loss.backward()
        self.optimizer.step()

        return res, train_cm_

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = 0.0
        count = 1.0
        val_cm = np.zeros((self.num_classes, self.num_classes))
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val'):
            self.optimizer.zero_grad()

            _, loss, val_res, val_cm_ = self.model_fn(self.model, data, eval=True)

            val_cm += val_cm_

            total_loss += loss.item()
            count += 1
            for k, v in val_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]

        return total_loss / count, eval_dict, val_cm

    def train(self, start_it, start_epoch, n_epochs, train_loader, test_loader=None, best_loss=0.0):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """
        eval_frequency = (self.eval_frequency if self.eval_frequency > 0 else len(train_loader))

        it = start_it
        with tqdm.trange(start_epoch, n_epochs + 1, desc='epochs') as tbar, \
                tqdm.tqdm(total=eval_frequency, leave=False, desc='train') as pbar:

            for epoch in tbar:
                self.logging.info("epoch: %03d", epoch)
                train_cm = np.zeros((self.num_classes, self.num_classes))
                for batch in train_loader:
                    res, train_cm_ = self._train_it(it, batch)
                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if self.viz is not None:
                        self.viz.update('train', it, res)

                    train_cm += train_cm_

                    if (it % eval_frequency) == 0:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, val_res, val_cm = self.eval_epoch(test_loader)

                            val_oa = overall_accuracy(val_cm)
                            val_macc, val_acc = accuracy_per_class(val_cm)
                            val_miou, val_iou = iou_per_class(val_cm)
                            val_mf1, val_f1 = f1score_per_class(val_cm)

                            self.logging.info("val_OA: %f", val_oa)
                            self.logging.info("val_mACC: %f", val_macc)
                            self.logging.info("val_mIoU: %f", val_miou)
                            self.logging.info("val_mF1: %f", val_mf1)

                            if self.viz is not None:
                                self.viz.update('val', it, val_res)

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(checkpoint_state(self.model, self.optimizer, val_loss, epoch, it),
                                            is_best, filename=self.checkpoint_name, bestname=self.best_name)

                        pbar = tqdm.tqdm(total=eval_frequency, leave=False, desc='train')
                        pbar.set_postfix(dict(total_it=it))

                    self.viz.flush()

                train_oa = overall_accuracy(train_cm)
                train_macc, train_acc = accuracy_per_class(train_cm)
                train_miou, train_iou = iou_per_class(train_cm)
                train_mf1, train_f1 = f1score_per_class(train_cm)

                self.logging.info("train_OA: %f", train_oa)
                self.logging.info("train_mACC: %f", train_macc)
                self.logging.info("train_mIoU: %f", train_miou)
                self.logging.info("train_mF1: %f", train_mf1)

        return best_loss
