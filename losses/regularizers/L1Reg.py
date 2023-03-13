import torch
from torch import nn


class Regularizer(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.name = "L1Reg"
        self.model = model
        self.config = config
        self.l1_reg_lambda = config.MODEL.OFA.REGULARIZER_L1_LAMBDA
        self.warmup_epochs = config.MODEL.OFA.REGULARIZER_WARMUP_EPOCHS

    def cal_lambda(self, epoch):
        # i: index 0,1,2,3,...,decay_bins
        # i=0, first channel; i=decay_bins-1, last channel
        if epoch<self.warmup_epochs: # normal regularizer during warmup
            return 0 # we don't use L1 normalization at first
        return self.l1_reg_lambda

    def __call__(self, width_mult=1.0, epoch=-1):
        loss = 0.
        #for n, p in self.model.named_parameters():
        for n, p in self.model.named_parameters():
            #print(n)
            if 'encoder_k' in n or 'proj_conv' in n: # skip encoder_k
                continue
            length = len(p.size())
            if length == 1: # bn weight/bias, group L2
                loss += self.cal_lambda(epoch) * torch.abs(p).sum()
        return loss
