import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from losses import BaseLoss as Base

class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "MSELoss"
        self.criterion = nn.MSELoss()

    def __call__(self, student_output, teacher_output, epoch=None, width_index=999):
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
                        #layer_number = int(feat_name.split('.')[0][-1])
                        #print(feat_name, layer_number)
                        #if width_index==1 and layer_number < 4: 
                        #if layer_number <= 4 - width_index: # for width_index=1, only layer4 used
                        #    continue
                        feat_t = q[feat_name]['output']
                        feat_s = student_output[v][feat_name]['output']
                        loss = self.criterion(feat_s, feat_t.detach())
                        #loss = torch.mean(torch.square(student_output[v]-q.detach()))
                        total_loss += loss.mean()
                        n_loss_terms += 1
                else:
                    loss = self.criterion(student_output[v], q.detach())
                    total_loss += loss.mean()
                    n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss