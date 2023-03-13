import torch
from torch import nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from losses import BaseLoss as Base


class Criterion(Base):
    def __init__(self, config):
        super(Criterion, self).__init__(config)
        self.name = "CE"
        self.criterion = torch.nn.CrossEntropyLoss()
        if config.AUG.TRAIN.MIX.PROB:
            self.criterion = SoftTargetCrossEntropy()
        elif config.TRAIN.LABEL_SMOOTHING > 0:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=config.TRAIN.LABEL_SMOOTHING
            )

    def __call__(self, pred, target):
        val = self.criterion(pred, target)
        self.update(val.item(), batch_size=pred.shape[0])
        return val
