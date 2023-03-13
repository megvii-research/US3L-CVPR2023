import torch
import torch.nn as nn
from copy import deepcopy
from utils.forward_hook import ForwardHookManager
from models.OFA.slimmable_ops import USConv2d

class SimCLR(nn.Module):
    """
    Build a SimCLR model 
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 128)
         """
        super(SimCLR, self).__init__()

        self.config = config
        dim = config.SSL.SETTING.DIM
        mlp = config.SSL.SETTING.MLP
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM
        with_predictor = config.SSL.SETTING.PREDICTOR

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=hidden_dim)

        prev_dim = self.encoder_q.fc.weight.shape[1]

        fc_dim = hidden_dim

        if mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim,  dim))

        # build a 2-layer predictor
        if with_predictor:
            self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(hidden_dim, dim)) # output layer
        else:
            self.predictor = nn.Identity()

    def forward(self, x1, x2, momentum_update=True):
        f1 = self.encoder_q(x1)  # queries: NxC
        #f1 = nn.functional.normalize(f1, dim=1)

        f2 = self.encoder_q(x2)  # queries: NxC
        #f2 = nn.functional.normalize(f2, dim=1)

        if not momentum_update:
            f1 = self.predictor(f1)
            f2 = self.predictor(f2)

        return [f1, f2], [f1.detach(), f2.detach()]


class SimCLR_Momentum(nn.Module):
    """
    Build a SimCLR model 
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 128)
         """
        super(SimCLR_Momentum, self).__init__()

        self.config = config
        dim = config.SSL.SETTING.DIM
        mlp = config.SSL.SETTING.MLP
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM
        self.m = config.SSL.SETTING.MOMENTUM
        with_predictor = config.SSL.SETTING.PREDICTOR

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=hidden_dim)

        prev_dim = self.encoder_q.fc.weight.shape[1]

        fc_dim = hidden_dim

        if mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim,  dim))

        # build a 2-layer predictor
        if with_predictor:
            self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(hidden_dim, dim)) # output layer
        else:
            self.predictor = nn.Identity()
        
        self.encoder_k  = deepcopy(self.encoder_q)
        self.encoder_k.apply(lambda m: setattr(m, 'width_mult', 1.0))
        if config.MODEL.OFA.DISTILL_FEATURE:
            #forward hook to extract intermediate features, both q and k
            device =  torch.device(config.DEVICE)
            self.forward_hook_manager_q = ForwardHookManager(device)
            self.forward_hook_manager_k = ForwardHookManager(device)
            #forward_hook_manager.add_hook(model, 'conv1', requires_input=True, requires_output=False)
            self.feat_name_list = config.MODEL.OFA.DISTILL_FEATURE_NAME

            self.proj_conv = nn.ModuleDict()
            
            for feat_name, feat_dim in zip(self.feat_name_list, config.MODEL.OFA.DISTILL_FEATURE_DIM):
                self.forward_hook_manager_q.add_hook(self.encoder_q, feat_name, requires_input=False, requires_output=True)
                self.forward_hook_manager_k.add_hook(self.encoder_k, feat_name, requires_input=False, requires_output=True)

                # linear conv
                self.proj_conv[feat_name.replace('.','_')] = USConv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False, us=[True, False])

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def forward_proj_conv(self, x):
        return self.proj_conv(x)

    def forward(self, x1, x2, ret_q_feature=False, ret_k_feature=False, proj_conv=False, momentum_update=True):
        int_feat_q1, int_feat_q2, int_feat_k1, int_feat_k2 = None, None, None, None
        
        f1 = self.encoder_q(x1)  # queries: NxC
        #f1 = nn.functional.normalize(f1, dim=1)

        if ret_q_feature:
            int_feat_q1 = self.forward_hook_manager_q.pop_io_dict()

        f2 = self.encoder_q(x2)  # queries: NxC
        #f2 = nn.functional.normalize(f2, dim=1)

        if ret_q_feature:
            int_feat_q2 = self.forward_hook_manager_q.pop_io_dict()

        if momentum_update: # actually we only need to forward momentum teacher for the largest model once
            with torch.no_grad():
                self._momentum_update_key_encoder()  # update the key encoder
                f1_teacher = self.encoder_k(x1)

                if ret_k_feature:
                    int_feat_k1 = self.forward_hook_manager_k.pop_io_dict()
                    #int_feat_k1 = io_dict[self.feat_name]['output']

                f2_teacher = self.encoder_k(x2)

                if ret_k_feature:
                    int_feat_k2 = self.forward_hook_manager_k.pop_io_dict()
                    #int_feat_k2 = io_dict[self.feat_name]['output']
        else: # otherwise we fill empty values
            f1_teacher = f1  # hack: need to detach, None
            f2_teacher = f2  # hack: need to detach, None
            
            # we need to forward predictor for student
            f1 = self.predictor(f1)
            f2 = self.predictor(f2)

        if ret_q_feature or ret_k_feature:
            if proj_conv:
                for feat_name in self.feat_name_list:
                    int_feat_q1[feat_name]['output'] = self.proj_conv[feat_name.replace('.','_')](int_feat_q1[feat_name]['output'])
                    int_feat_q2[feat_name]['output'] = self.proj_conv[feat_name.replace('.','_')](int_feat_q2[feat_name]['output'])
            return [f1, f2], [f1_teacher.detach(), f2_teacher.detach()], [int_feat_q1, int_feat_q2], [int_feat_k1, int_feat_k2]


        return [f1, f2], [f1_teacher.detach(), f2_teacher.detach()]