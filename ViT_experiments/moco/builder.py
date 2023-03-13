# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
#from config import update_config


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        #self.config = config

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

class MoCo_USViT(MoCo_ViT):
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        super(MoCo_USViT, self).__init__(base_encoder, dim, mlp_dim, T)
        self.predictor_new = deepcopy(self.predictor)
        self.momentum_encoder.apply(lambda m: setattr(m, 'width_mult', 1.0))

    def forward(self, x1, x2, m, update=True, teacher_target=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        z1 = self.base_encoder(x1)
        z2 = self.base_encoder(x2)

        if update: # actually we only need to forward momentum teacher for the largest model once
            q1 = self.predictor(z1)
            q2 = self.predictor(z2)
            with torch.no_grad():
                self._update_momentum_encoder(m)  # update the momentum encoder
                k1 = self.momentum_encoder(x1)
                k2 = self.momentum_encoder(x2)
        else: # otherwise we fill empty values
            q1 = self.predictor_new(z1)
            q2 = self.predictor_new(z2)
            k1 = z1  # hack: need to detach, None
            k2 = z2  # hack: need to detach, None
        
        if teacher_target is not None:
            k1, k2 = teacher_target
        
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), [q1, q2], [k1.detach(), k2.detach()]

class MoCo_USBaselineViT(MoCo_ViT):
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        super(MoCo_USBaselineViT, self).__init__(base_encoder, dim, mlp_dim, T)
        #self.predictor_new = deepcopy(self.predictor)
        self.momentum_encoder.apply(lambda m: setattr(m, 'width_mult', 1.0))

    def forward(self, x1, x2, m, update=True, teacher_target=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        z1 = self.base_encoder(x1)
        z2 = self.base_encoder(x2)

        q1 = self.predictor(z1)
        q2 = self.predictor(z2)

        with torch.no_grad():
            self._update_momentum_encoder(m)  # update the momentum encoder
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        if teacher_target is not None:
            k1, k2 = teacher_target

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1), [q1, q2], [k1.detach(), k2.detach()]



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
