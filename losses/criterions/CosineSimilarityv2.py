import torch
from torch import nn
import torch.nn.functional as F
from losses import BaseLoss as Base

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class Criterion(Base):
    def __init__(self, config):
        super(Criterion, self).__init__(config)
        self.name = "CosineSimilarityv2"
        self.criterion = nn.CosineSimilarity(dim=1)
        #self.criterion = loss_fn

    def __call__(self, pred, target, epoch=None):
        p1, p2 = pred
        z1,  z2 = target
        
        #z = (z1+z2)/2 # cal average view target

        #val = (self.criterion(p1,z.detach())+self.criterion(p2,z.detach())).mean()
        val = (self.criterion(p1,z2.detach())+self.criterion(p2,z1.detach())).mean()
        self.update(val.item(), batch_size=p1.shape[0])
        return val
