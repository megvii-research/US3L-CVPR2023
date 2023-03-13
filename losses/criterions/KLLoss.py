import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from losses import BaseLoss as Base

class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "DINOLoss"

        self.student_temp = config.MODEL.OFA.DISTILL_STUDENT_TEMP
        self.teacher_temp = config.MODEL.OFA.DISTILL_TEACHER_TEMP

    def __call__(self, student_output, teacher_output, epoch=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_output):
            for v in range(len(student_output)):
                #if v != iq: # we skip cases where student and teacher operate on different views
                if v == iq: # we skip cases where student and teacher operate on the same view
                    continue
                # if q is a dict, it means we use feature distillation
                if isinstance(q, dict):
                    for feat_name in q.keys():
                        feat_t = q[feat_name]['output']
                        feat_s = student_output[v][feat_name]['output']
                        feat_t = torch.sum(feat_t, dim=1)
                        feat_s = torch.sum(feat_s, dim=1) 
                        feat_t = feat_t.reshape(feat_t.shape[0], -1) # N, HxW
                        feat_s = feat_s.reshape(feat_s.shape[0], -1) # N, HxW

                        feat_t = F.softmax(feat_t  / self.teacher_temp, dim=-1)
                        feat_t = feat_t.detach()
                        loss = torch.sum(-feat_t * F.log_softmax(feat_s / self.student_temp, dim=-1), dim=-1)
                        #loss = self.criterion(feat_s, feat_t.detach())
                        #loss = torch.mean(torch.square(student_output[v]-q.detach()))
                        total_loss += loss.mean()
                        n_loss_terms += 1
                else:
                    q = F.softmax(q  / self.teacher_temp, dim=-1)
                    loss = torch.sum(-q * F.log_softmax(student_output[v]/ self.student_temp, dim=-1), dim=-1)
                    total_loss += loss.mean()
                    n_loss_terms += 1
                
        total_loss /= n_loss_terms
        return total_loss

        '''
        student_output = student_output / self.student_temp
        # teacher centering and sharpening
        teacher_output = F.softmax(teacher_output  / self.teacher_temp, dim=-1)
        loss = torch.sum(-teacher_output.detach() * F.log_softmax(student_output, dim=-1), dim=-1).mean()
        return loss
        '''