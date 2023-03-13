from turtle import width
import torch
import torch.nn as nn

width_mult_list = [1.0, 0.75, 0.5, 0.25]

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(SwitchableBatchNorm2d, self).__init__()
        self.width_mult_list = width_mult_list
        
        self.num_features_list = []
        for width_mult in self.width_mult_list:
            self.num_features_list.append(int(num_features*width_mult))

        self.num_features = max(self.num_features_list)
        bns = []
        for i in self.num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(self.width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, us=[True, True]):
        super(SlimmableConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

        self.width_mult_list = width_mult_list
        self.in_channels_list = []
        self.out_channels_list = []
        for width_mult in self.width_mult_list:
            if us[0]:
                self.in_channels_list.append(int(in_channels*width_mult))
            else:
                self.in_channels_list.append(in_channels)
            
            if us[1]:
                self.out_channels_list.append(int(out_channels*width_mult))
            else:
                self.out_channels_list.append(out_channels)

        #self.groups_list = groups_list
        if groups==1:
            self.groups_list = [1 for _ in range(len(self.in_channels_list))]
        else:
            self.groups_list = [int(groups*width_mult) for width_mult in self.width_mult_list]
        
        self.width_mult = max(self.width_mult_list)

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        super(SlimmableLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.width_mult_list = width_mult_list
        self.in_features_list = []
        self.out_features_list = []
        for width_mult in self.width_mult_list:
            if us[0]:
                self.in_features_list.append(int(in_features*width_mult))
            else:
                self.in_features_list.append(in_features)

            if us[1]:           
                self.out_features_list.append(int(out_features*width_mult))
            else:
                self.out_features_list.append(out_features)

        self.width_mult = max(self.width_mult_list)

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        #if getattr(FLAGS, 'conv_averaged', False):
        #    y = y * (max(self.in_channels_list) / self.in_channels)
        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.width_mult = None
        self.us = us

    def forward(self, input):
        if self.us[0]:
            self.in_features = make_divisible(
                self.in_features_max * self.width_mult)
        if self.us[1]:
            self.out_features = make_divisible(
                self.out_features_max * self.width_mult)
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        
        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1, track_running_stats=False):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=track_running_stats)
        self.num_features_max = num_features
        # for tracking performance during training
        #self.width_mult_list = []
        #self.width_mult_list = [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
        #self.width_mult_list = FLAGS.MODEL.OFA.WIDTH_MULT_LIST
        #self.bn = nn.ModuleList([
        #    nn.BatchNorm2d(i, affine=False) for i in [
        #        make_divisible(
        #            self.num_features_max * width_mult / ratio) * ratio
        #        for width_mult in self.width_mult_list]])
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):

        weight = self.weight
        bias = self.bias
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        
        y = nn.functional.batch_norm(
            input,
            self.running_mean[:c] if self.track_running_stats else self.running_mean,
            self.running_var[:c] if self.track_running_stats else self.running_var,
            weight[:c],
            bias[:c],
            self.training,
            self.momentum,
            self.eps)
        return y


class USBatchNorm2dv2(nn.BatchNorm2d): # different width for different channels
    def __init__(self, num_features, ratio=1, track_running_stats=False):
        super(USBatchNorm2dv2, self).__init__(
            num_features, affine=True, track_running_stats=track_running_stats)
        self.num_features_max = num_features
        self.ratio = ratio
        self.width_mult = None
        self.register_buffer("mask", None)
        self.ignore_model_profiling = True
    
    def cal_pruned_mask(self, ):
        self.mask = torch.ones_like(self.weight)
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        self.mask[c:] = 0
        self.mask = self.mask.reshape(1, -1, 1, 1)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        
        y = nn.functional.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            weight,
            bias,
            self.training,
            self.momentum,
            self.eps)
        return y * self.mask


class PrunedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True):
        super(PrunedConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        #self.w_mask = None
        #self.b_mask = None
        self.register_buffer("w_mask", None)
        self.register_buffer("b_mask", None)
        self.register_buffer("sort_indices", None)

    def cal_pruned_mask(self, config, resort=True):
        if config.PRUNE.STRATEGY == "l1norm":
            def sort_indices(data):
                sum_data = torch.sum(torch.abs(data.reshape(data.shape[0], -1)), dim=1)
                sorted_sum_weight, indices = torch.sort(sum_data, dim=0)
                #return 
                return indices
            def get_prune_mask(data, indices):
                dim = 0  # default is channelwise
                mask = torch.ones_like(data)
                for idx in indices:
                    mask[idx] = torch.zeros_like(
                        torch.index_select(data.cpu().detach(), dim, torch.tensor(idx))
                    )
                return mask

            oc = self.weight.data.shape[0]
            num_elimination = int(oc * (1-config.PRUNE.GLOBAL_FACTOR))

            if self.sort_indices is None or resort:
                self.sort_indices = sort_indices(self.weight)

            indices = self.sort_indices[:num_elimination].tolist()
            self.w_mask = get_prune_mask(self.weight.data, indices)
            if self.bias is not None:
                self.b_mask = get_prune_mask(self.bias.data, indices)
        else:
            print('Not supported')

        return indices

    def forward(self, input):
        #weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        weight = self.weight * self.w_mask

        if self.bias is not None:
            bias = self.bias * self.b_mask
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        #if getattr(FLAGS, 'conv_averaged', False):
        #    y = y * (max(self.in_channels_list) / self.in_channels)
        return y


class PrunedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1, track_running_stats=False):
        super(PrunedBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=track_running_stats)
        self.num_features_max = num_features
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True
        self.register_buffer("mask", None)

    def set_pruned_mask(self, indices):
        def get_prune_mask(data, indices):
                dim = 0  # default is channelwise
                mask = torch.ones_like(data)
                for idx in indices:
                    mask[idx] = torch.zeros_like(
                        torch.index_select(data.cpu().detach(), dim, torch.tensor(idx))
                    )
                return mask

        self.mask = get_prune_mask(self.weight, indices)
        self.mask = self.mask.reshape(1, -1, 1, 1)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        
        y = nn.functional.batch_norm(
            input,
            self.running_mean if self.track_running_stats else self.running_mean,
            self.running_var if self.track_running_stats else self.running_var,
            weight,
            bias,
            self.training,
            self.momentum,
            self.eps)
        y = y * self.mask
        return y


class PrunedBatchNorm2dv2(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1, track_running_stats=False):
        super(PrunedBatchNorm2dv2, self).__init__(
            num_features, affine=True, track_running_stats=track_running_stats)
        self.num_features_max = num_features
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True
        self.register_buffer("mask", None)

    def cal_pruned_mask(self, config):
        if config.PRUNE.STRATEGY == "l1norm":
            def sort_indices(data):
                sum_data = torch.sum(torch.abs(data.reshape(data.shape[0], -1)), dim=1)
                sorted_sum_weight, indices = torch.sort(sum_data, dim=0)
                return indices
            def get_prune_mask(data, indices):
                dim = 0  # default is channelwise
                mask = torch.ones_like(data)
                for idx in indices:
                    mask[idx] = torch.zeros_like(
                        torch.index_select(data.cpu().detach(), dim, torch.tensor(idx))
                    )
                return mask

            oc = self.weight.data.shape[0]
            num_elimination = int(oc * (1-config.PRUNE.GLOBAL_FACTOR))

            indices = sort_indices(self.weight)
            indices = indices[:num_elimination].tolist()
            self.mask = get_prune_mask(self.weight.data, indices)         
            self.mask = self.mask.reshape(1, -1, 1, 1) 
        else:
            print('Not supported')

        return indices

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        
        y = nn.functional.batch_norm(
            input,
            self.running_mean if self.track_running_stats else self.running_mean,
            self.running_var if self.track_running_stats else self.running_var,
            weight,
            bias,
            self.training,
            self.momentum,
            self.eps)
        y = y * self.mask
        return y

def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]


def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        if getattr(FLAGS, 'MODEL.OFA.CUMULATIVE_BN_STATS', False):
            m.momentum = None