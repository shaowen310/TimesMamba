import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def get_lr_scheduler(optimizer, train_epochs, warmup_epochs=0, lradj="type1"):
    schedulers = []
    lr_sched_milestones = []

    if warmup_epochs > 0:
        warmup_fn = lambda c: 1 / (10 ** (float(warmup_epochs - c)))
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
        schedulers.append(warmup_scheduler)
        lr_sched_milestones.append(warmup_epochs)

    if lradj == "type1":
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, train_epochs - warmup_epochs, eta_min=1e-5
        )
        schedulers.append(cosine_scheduler)
    elif lradj == "type2":
        exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
        schedulers.append(exp_scheduler)
    else:
        ms_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2, 4, 6, 8, 10, 15, 20], gamma=0.5
        )
        schedulers.append(ms_scheduler)

    return optim.lr_scheduler.SequentialLR(optimizer, schedulers, lr_sched_milestones)


class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.patience >= 0 and self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = min(score, self.best_score)
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = min(val_loss, self.val_loss_min)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
