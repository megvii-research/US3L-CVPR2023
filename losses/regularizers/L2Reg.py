import torch
from torch import nn


class Regularizer(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.name = "GroupReg"
        self.model = model
        self.config = config
        self.reg_lambda = config.MODEL.OFA.REGULARIZER_LAMBDA

    def __call__(self, width_mult=1.0, epoch=-1):
        loss = 0.
        #for n, p in self.model.named_parameters():
        for n, p in self.model.named_parameters():
            #print(n)
            if 'encoder_k' in n or 'proj_conv' in n: # skip encoder_k
                continue
            
            loss += self.reg_lambda * torch.square(p).sum()
        return loss
