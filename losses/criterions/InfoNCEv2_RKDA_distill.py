import torch
from torch import nn
import torch.nn.functional as F
from losses import BaseLoss as Base
from .RKDALoss import RKdAngle

def nt_xentv2(f1, f2, t=0.5):
    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    x = torch.cat([f1,f2], dim=0)
    batch_size = f1.size(0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(x, x.t().contiguous()) / t)

    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(f1 *f2, dim=-1) / t)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

class Criterion(Base):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, config):
        super(Criterion, self).__init__(config)
        self.name = "SimCLR"
        self.temperature = config.SSL.SETTING.T
        self.criterion = RKdAngle()
    
    def __call__(self, f_student, f_teacher, epoch=-1):
        total_loss = 0
        n_loss_terms = 0

        for s in range(len(f_student)):
            for t in range(len(f_teacher)):
                if s == t:
                    # we skip cases where student and teacher operate on the same view
                    continue

                loss1 = nt_xentv2(f_student[s], f_teacher[t].detach(), self.temperature)
                total_loss += loss1.mean()
                loss2 = self.criterion(f_student[s], f_teacher[t].detach())
                total_loss += loss2.mean()
                
                n_loss_terms += 1
                bs = f_student[s].shape[0]
        total_loss /= n_loss_terms
        self.update(total_loss.item(), batch_size=bs)
        return total_loss