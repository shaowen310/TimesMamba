import torch

import loggingutil
from model import (
    TimesMamba,
)

logger = loggingutil.get_logger(__name__)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "TimesMamba": TimesMamba,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device("cuda:{}".format(self.args.device))
            print("Use GPU: cuda:{}".format(self.args.device))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
