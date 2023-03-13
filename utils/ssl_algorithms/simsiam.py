import torch
import torch.nn as nn
from copy import deepcopy
from models.OFA.slimmable_ops import USConv2d
from utils.forward_hook import ForwardHookManager

'''
class SimSiam(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.config = config
        
        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder_q.fc.weight.shape[1]

        # to make resnet-18 and resnet-34 have larger hidden dimensions, e.g., 2048
        fc_dim = prev_dim
        fc_dim = max(prev_dim, dim)
        fc_dim = max(hidden_dim, fc_dim)
        self.encoder_q.fc = nn.Sequential(nn.Linear(prev_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim,  dim),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer


        self.encoder_q.fc[6].bias.requires_grad = False #hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        z1 = self.encoder_q(x1)
        z2 = self.encoder_q(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return [p1, p2] , [z1.detach(), z2.detach()]
'''

class SimSiam(nn.Module):
    """
    Build a SimSiam model, compatible to USNets.
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.config = config
        
        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=hidden_dim, bn_track_stats=True)

        fc_dim = hidden_dim

        self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim,  dim),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

        self.encoder_q.fc[6].bias.requires_grad = False #hack: not use bias as it is followed by BN
        #self.projector = nn.Sequential(
        #                                nn.BatchNorm1d(fc_dim),
        #                                nn.ReLU(inplace=True), # first layer
        #                                nn.Linear(fc_dim, fc_dim, bias=False),
        #                                nn.BatchNorm1d(fc_dim),
        #                                nn.ReLU(inplace=True), # second layer
        #                                nn.Linear(fc_dim,  dim),
        #                                nn.BatchNorm1d(dim, affine=False)) # output layer

        #self.projector[5].bias.requires_grad = False #hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, dim)) # output layer

    def forward(self, x1, x2, momentum_update=True):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        #z1 = self.projector(self.encoder_q(x1))
        #z2 = self.projector(self.encoder_q(x2))
        z1 = self.encoder_q(x1)
        z2 = self.encoder_q(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return [p1, p2], [z1.detach(), z2.detach()]

class SimSiam_Momentum(nn.Module):
    """
    Build a SimSiam model, compatible to USNets.
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam_Momentum, self).__init__()
        self.config = config
        
        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM
        self.m = config.SSL.SETTING.MOMENTUM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=hidden_dim, zero_init_residual=True)

        fc_dim = hidden_dim

        self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim,  dim),
                                        ) # output layer

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, dim)) # output layer


        # build a 2-layer predictor: additional for distillation
        if config.SSL.SETTING.NEW_DISTILL_HEAD:
            self.predictor_new = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(hidden_dim, dim)) # output layer
        else:
            self.predictor_new = self.predictor

        self.encoder_k  = deepcopy(self.encoder_q)
        self.encoder_k.apply(lambda m: setattr(m, 'width_mult', 1.0))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2, momentum_update=True):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        #z1 = self.projector(self.encoder_q(x1))
        #z2 = self.projector(self.encoder_q(x2))
        z1 = self.encoder_q(x1)
        z2 = self.encoder_q(x2)

        #p1 = self.predictor(z1)
        #p2 = self.predictor(z2)

        if momentum_update: # actually we only need to forward momentum teacher for the largest model once
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            with torch.no_grad():
                self._momentum_update_key_encoder()  # update the key encoder
                teacher_z1 = self.encoder_k(x1)
                teacher_z2 = self.encoder_k(x2)
        else: # otherwise we fill empty values
            p1 = self.predictor_new(z1)
            p2 = self.predictor_new(z2)
            teacher_z1 = z1  # hack: need to detach, None
            teacher_z2 = z2  # hack: need to detach, None
        
        return [p1, p2], [teacher_z1.detach(), teacher_z2.detach()] # asymmertric distill head

        if momentum_update:
            #return [p1, p2], [z1.detach(), z2.detach()], [z1.detach(), z2.detach()]
            return [p1, p2], [teacher_z1.detach(), teacher_z2.detach()] # asymmertric distill head
            #return [p1, p2], [z1.detach(), z2.detach()], [teacher_z1.detach(), teacher_z2.detach()]
            #return [p1, p2], [teacher_z1.detach(), teacher_z2.detach()], [teacher_z1.detach(), teacher_z2.detach()] # max with momentum
        else:
            #return [z1, z2], [teacher_z1.detach(), teacher_z2.detach()]
            return [p1, p2], [teacher_z1.detach(), teacher_z2.detach()] # asymmertric distill head
