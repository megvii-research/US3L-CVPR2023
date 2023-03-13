import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "DINOLoss"
        self.student_temp = config.MODEL.OFA.DISTILL_STUDENT_TEMP
        self.center_momentum = 0.9 # hack

        out_dim = config.SSL.SETTING.DIM
        self.register_buffer("center", torch.zeros(1, out_dim))

        warmup_teacher_temp = 0.04
        warmup_teacher_temp_epochs = 30
        nepochs = config.TRAIN.EPOCHS
        teacher_temp = config.MODEL.OFA.DISTILL_TEACHER_TEMP
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def __call__(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        #student_out = student_output / self.student_temp
        #student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        #teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        #teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_output):
            for v in range(len(student_output)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                q = F.softmax((q - self.center) / temp, dim=-1)
                loss = torch.sum(-q * F.log_softmax(student_output[v]/ self.student_temp, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        teacher_output = torch.stack(teacher_output, dim=0)
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #batch_center = torch.sum(teacher_output)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)