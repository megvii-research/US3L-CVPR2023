from shlex import quote
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
        self.name = "CosineSimilarity"
        self.criterion = nn.CosineSimilarity(dim=1)

    def __call__(self, pred, target, epoch=None):
        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(target):
            for v in range(len(pred)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                # if q is a dict, it means we use feature distillation
                if isinstance(q, dict):
                    for feat_name in q.keys():
                        feat_t = q[feat_name]['output']
                        feat_s = pred[v][feat_name]['output']
                        # pixel level
                        #feat_t = torch.flatten(feat_t, start_dim=1)
                        #feat_s = torch.flatten(feat_s, start_dim=1)
                        # GAP
                        feat_t = torch.sum(feat_t, dim=(2,3))
                        feat_s = torch.sum(feat_s, dim=(2,3))
                        #print(feat_s.size(), feat_t.size())
                        loss = -self.criterion(feat_s, feat_t.detach())
                        #loss = torch.mean(torch.square(student_output[v]-q.detach()))
                        total_loss += loss.mean()
                        n_loss_terms += 1
                        bs = feat_t.shape[0]
                else:
                    loss = -self.criterion(pred[v], q.detach())
                    total_loss += loss.mean()
                    n_loss_terms += 1
                    bs = q.shape[0]
        total_loss /= n_loss_terms
        self.update(total_loss.item(), batch_size=bs)
        return total_loss

'''
class Criterion(Base):
    def __init__(self, config, model):
        super(Criterion, self).__init__(config, model)
        self.name = "CosineSimilarity"
        self.criterion = nn.CosineSimilarity(dim=1)

    def __call__(self, pred, target, epoch=None):
        if len(pred)==2:
            p1, p2 = pred
            if len(target)==2:
                z1,  z2 = target
                val = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
            else:
                z1, z2, oz1, oz2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(p1, oz2.detach()).mean() + self.criterion(p2, oz1.detach()).mean()) * 0.5
                val = (val1+val2)*0.5
        else:
            p1, p2, op1, op2 = pred
            if len(target)==2:
                z1, z2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(op1, z2.detach()).mean() + self.criterion(op2, z1.detach()).mean()) * 0.5
                val = (val1+val2)*0.5
            else:
                z1, z2, oz1, oz2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(op1, z2.detach()).mean() + self.criterion(op2, z1.detach()).mean()) * 0.5
                val3 = -(self.criterion(op1, oz2.detach()).mean() + self.criterion(op2, oz1.detach()).mean()) * 0.5
                val = (val1+val2+val3)/3

        self.update(val.item(), batch_size=p1.shape[0])
        return val
'''