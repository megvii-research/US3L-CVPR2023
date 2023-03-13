import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from losses import BaseLoss as Base

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "RKDD"
        self.criterion = RkdDistance()

    def __call__(self, student_output, teacher_output, epoch=None):
        """
        Relation knowledge distillation of the teacher and student networks.
        """

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_output):
            for v in range(len(student_output)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                #q = F.softmax((teacher_output - self.center) / temp, dim=-1)
                if isinstance(q, dict):
                    for feat_name in q.keys():
                        feat_t = q[feat_name]['output']
                        feat_s = student_output[v][feat_name]['output']
                        feat_t = torch.flatten(feat_t, start_dim=1)
                        feat_s = torch.flatten(feat_s, start_dim=1)
                
                        loss = self.criterion(feat_s, feat_t.detach())
                        total_loss += loss.mean()
                        n_loss_terms += 1
                else:
                    loss = self.criterion(student_output[v], q)
                    total_loss += loss.mean()
                    n_loss_terms += 1
                
        total_loss /= n_loss_terms
        return total_loss