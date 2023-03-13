import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from losses import BaseLoss as Base


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "RKDAMSE"
        self.criterion = RKdAngle()
        self.criterion2 = nn.CosineSimilarity(dim=1)


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
                        n,c,h,w = feat_s.size()
                        feat_t = torch.flatten(feat_t, start_dim=1)
                        feat_s = torch.flatten(feat_s, start_dim=1)
                
                        loss = self.criterion(feat_s, feat_t.detach())
                        total_loss += loss.mean()
                        n_loss_terms += 1
                else:
                    loss1 = self.criterion(student_output[v], q.detach())
                    total_loss += loss1.mean()

                    loss2 = -self.criterion(student_output[v], q.detach())
                    total_loss += loss2.mean()
                    n_loss_terms += 1

        total_loss /= n_loss_terms
        return total_loss