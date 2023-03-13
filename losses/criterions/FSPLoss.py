import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from losses import BaseLoss as Base

class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = "GramLoss"

    @staticmethod
    def compute_fsp_matrix(first_feature_map, second_feature_map):
        first_h, first_w = first_feature_map.shape[2:4]
        second_h, second_w = second_feature_map.shape[2:4]
        target_h, target_w = min(first_h, second_h), min(first_w, second_w)
        if first_h > target_h or first_w > target_w:
            first_feature_map = F.adaptive_max_pool2d(first_feature_map, (target_h, target_w))

        if second_h > target_h or second_w > target_w:
            second_feature_map = F.adaptive_max_pool2d(second_feature_map, (target_h, target_w))

        first_feature_map = first_feature_map.flatten(2)
        second_feature_map = second_feature_map.flatten(2)
        hw = first_feature_map.shape[2]
        return torch.matmul(first_feature_map, second_feature_map.transpose(1, 2)) / hw

    def __call__(self, student_output, teacher_output, epoch=None):
        fsp_loss = 0
        batch_size = None

        student_first_feature_map = student_output[0]
        student_second_feature_map = student_output[1]
        student_fsp_matrices = self.compute_fsp_matrix(student_first_feature_map, student_second_feature_map)
        teacher_first_feature_map = teacher_output[0]
        teacher_second_feature_map = teacher_output[1]
        teacher_fsp_matrices = self.compute_fsp_matrix(teacher_first_feature_map, teacher_second_feature_map)
        #print(student_fsp_matrices.size())
        fsp_loss += (student_fsp_matrices - teacher_fsp_matrices.detach()).norm(dim=1).mean()
        if batch_size is None:
            batch_size = student_first_feature_map.shape[0]
            
        return fsp_loss / batch_size