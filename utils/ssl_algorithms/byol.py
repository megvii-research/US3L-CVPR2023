from models.OFA.slimmable_ops import USConv2d
import torch
import torch.nn as nn
from copy import deepcopy
from utils.forward_hook import ForwardHookManager


class BYOL(nn.Module):
    """
    Build a BYOL model.
    """

    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(BYOL, self).__init__()

        self.config = config

        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM
        self.m = config.SSL.SETTING.MOMENTUM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        if config.MODEL.OFA.NUM_SAMPLE_TRAINING == 1:
            print('normal BN')
            bn_track_stats = True
        else:
            print('US BN')
            bn_track_stats = False
        self.encoder_q = base_encoder(num_classes=hidden_dim, bn_track_stats=bn_track_stats)

        print(self.encoder_q.state_dict().keys())
        print(self.encoder_q)
        # build a 3-layer projector
        fc_dim = hidden_dim

        self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                          nn.BatchNorm1d(fc_dim),
                                          nn.ReLU(inplace=True),  # first layer
                                          nn.Linear(fc_dim, fc_dim, bias=False),
                                          nn.BatchNorm1d(fc_dim),
                                          nn.ReLU(inplace=True),  # second layer
                                          nn.Linear(fc_dim, dim))  # output layer

        self.encoder_k = deepcopy(self.encoder_q)

        self.encoder_k.apply(lambda m: setattr(m, 'width_mult', 1.0))

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(hidden_dim, dim))  # output layer

        # build a 2-layer predictor: additional for distillation
        if config.SSL.SETTING.NEW_DISTILL_HEAD:
            self.predictor_new = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                               nn.BatchNorm1d(hidden_dim),
                                               nn.ReLU(inplace=True),  # hidden layer
                                               nn.Linear(hidden_dim, dim))  # output layer
        else:
            self.predictor_new = self.predictor

        if config.MODEL.OFA.DISTILL_FEATURE:
            # forward hook to extract intermediate features, both q and k
            device = torch.device(config.DEVICE)
            self.forward_hook_manager_q = ForwardHookManager(device)
            self.forward_hook_manager_k = ForwardHookManager(device)
            # forward_hook_manager.add_hook(model, 'conv1', requires_input=True, requires_output=False)
            self.feat_name_list = config.MODEL.OFA.DISTILL_FEATURE_NAME

            self.proj_conv = nn.ModuleDict()

            for feat_name, feat_dim in zip(self.feat_name_list, config.MODEL.OFA.DISTILL_FEATURE_DIM):
                self.forward_hook_manager_q.add_hook(self.encoder_q, feat_name, requires_input=False,
                                                     requires_output=True)
                self.forward_hook_manager_k.add_hook(self.encoder_k, feat_name, requires_input=False,
                                                     requires_output=True)

                # linear conv
                self.proj_conv[feat_name.replace('.', '_')] = USConv2d(feat_dim, feat_dim, kernel_size=1, stride=1,
                                                                       padding=0, bias=False, us=[True, False])
                # feat_dim = config.MODEL.OFA.DISTILL_FEATURE_DIM
                # projection conv
                # self.proj_conv = nn.Sequential(
                #            USConv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False, us=[True, False]),
                #            nn.BatchNorm2d(feat_dim),
                #            nn.ReLU(),
                #            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False)
                #    )

                # linear conv
                # self.proj_conv = USConv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False, us=[True, False])

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
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """

        int_feat_q1, int_feat_q2, int_feat_k1, int_feat_k2 = None, None, None, None

        online_proj_one = self.encoder_q(x1)
        # if use intermediate feature
        if ret_q_feature:
            int_feat_q1 = self.forward_hook_manager_q.pop_io_dict()
            # int_feat_q1 = io_dict[self.feat_name]['output']

        online_proj_two = self.encoder_q(x2)
        # if use intermediate feature
        if ret_q_feature:
            int_feat_q2 = self.forward_hook_manager_q.pop_io_dict()
            # int_feat_q2 = io_dict[self.feat_name]['output']

        if momentum_update:  # actually we only need to forward momentum teacher for the largest model once
            online_pred_one = self.predictor(online_proj_one)
            online_pred_two = self.predictor(online_proj_two)
            with torch.no_grad():
                self._momentum_update_key_encoder()  # update the key encoder
                target_proj_one = self.encoder_k(x1)
                if ret_k_feature:
                    int_feat_k1 = self.forward_hook_manager_k.pop_io_dict()
                    # int_feat_k1 = io_dict[self.feat_name]['output']

                target_proj_two = self.encoder_k(x2)
                if ret_k_feature:
                    int_feat_k2 = self.forward_hook_manager_k.pop_io_dict()
                    # int_feat_k2 = io_dict[self.feat_name]['output']
        else:  # otherwise we fill empty values
            online_pred_one = self.predictor_new(online_proj_one)
            online_pred_two = self.predictor_new(online_proj_two)

            target_proj_one = online_pred_one  # hack: need to detach, None
            target_proj_two = online_pred_two  # hack: need to detach, None

        if ret_q_feature or ret_k_feature:
            if proj_conv:
                for feat_name in self.feat_name_list:
                    int_feat_q1[feat_name]['output'] = self.proj_conv[feat_name.replace('.', '_')](
                        int_feat_q1[feat_name]['output'])
                    int_feat_q2[feat_name]['output'] = self.proj_conv[feat_name.replace('.', '_')](
                        int_feat_q2[feat_name]['output'])
            return [online_pred_one, online_pred_two], [target_proj_one.detach(), target_proj_two.detach()], [
                int_feat_q1, int_feat_q2], [int_feat_k1, int_feat_k2]

        return [online_pred_one, online_pred_two], [target_proj_one.detach(), target_proj_two.detach()]


class BYOL_InterFeat(nn.Module):
    """
    Build a BYOL model with intermediate feature distillation.
    """

    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(BYOL, self).__init__()

        self.config = config

        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM
        self.m = config.SSL.SETTING.MOMENTUM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=hidden_dim)

        # build a 3-layer projector
        fc_dim = hidden_dim

        self.encoder_q.fc = nn.Sequential(self.encoder_q.fc,
                                          nn.BatchNorm1d(fc_dim),
                                          nn.ReLU(inplace=True),  # first layer
                                          nn.Linear(fc_dim, fc_dim, bias=False),
                                          nn.BatchNorm1d(fc_dim),
                                          nn.ReLU(inplace=True),  # second layer
                                          nn.Linear(fc_dim, dim))  # output layer

        self.encoder_k = deepcopy(self.encoder_q)

        self.encoder_k.apply(lambda m: setattr(m, 'width_mult', 1.0))

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(hidden_dim, dim))  # output layer

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # print(param_q.size(), param_k.size())
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        online_proj_one = self.encoder_q(x1)
        online_proj_two = self.encoder_q(x2)

        online_pred_one = self.predictor(online_proj_one)
        online_pred_two = self.predictor(online_proj_two)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            target_proj_one = self.encoder_k(x1)
            target_proj_two = self.encoder_k(x2)

        return [online_pred_one, online_pred_two], [target_proj_one.detach(), target_proj_two.detach()]