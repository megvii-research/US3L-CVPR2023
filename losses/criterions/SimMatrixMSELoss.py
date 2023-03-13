import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from losses import BaseLoss as Base


class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "Similarity matrix"
        self.criterion = nn.CosineSimilarity(dim=1)

    def __call__(self, student_output, teacher_output, epoch=None):
        """
        Similarity matrix distillation of the teacher and student networks.
        """

        total_loss = 0

        s1, s2 = student_output
        s1, s2 = F.normalize(s1, dim=-1, p=2), F.normalize(s2, dim=-1, p=2)

        t1, t2 = teacher_output
        t1, t2 = F.normalize(t1, dim=-1, p=2).detach(), F.normalize(t2, dim=-1, p=2).detach()

        #sim1 = torch.mm(s1, t2.T) # NxN
        simS = torch.mm(s1, s2.T) # NxN
        #sim3 = torch.mm(s2, t1.T) # NxN
        simT = torch.mm(t1, t2.T) # NxN

        #total_loss = ((sim1-simT).pow_(2).sum()+(sim2-simT).pow_(2).sum()+(sim3-simT.T).pow_(2).sum())/3 # 3 terms
        #total_loss = ((sim1-simT).pow_(2).sum()+(sim2-simT).pow_(2).sum())/2 # 2 terms

        #total_loss = (sim2-simT).pow_(2).sum()
        #print(total_loss.item())
        total_loss = 50.0*F.smooth_l1_loss(simS, simT, reduction='elementwise_mean').mean()
        #print(total_loss.item())

        total_loss += -0.5*(self.criterion(s1, t2)+self.criterion(s2, t1)).mean()


        return total_loss