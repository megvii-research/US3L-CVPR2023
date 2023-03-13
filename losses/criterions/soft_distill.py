import torch
from torch import nn
import torch.nn.functional as F
from losses import BaseLoss as Base

def soft_cross_entropy(student_logit, teacher_logit):
    '''
    :param student_logit: logit of the student arch (without softmax norm)
    :param teacher_logit: logit of the teacher arch (already softmax norm)
    :return: CE loss value.
    '''
    return -(teacher_logit * torch.nn.functional.log_softmax(student_logit, 1)).sum()/student_logit.shape[0]


class Criterion(Base):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, config):
        super(Criterion, self).__init__(config)
        self.name = "Soft KL"

    def __call__(self, logits_stu, logits_tea, epoch=-1):
        total_loss = soft_cross_entropy(logits_stu, logits_tea)
        return total_loss