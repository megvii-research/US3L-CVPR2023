# !/usr/bin/env python

import torch.nn as nn
import math
import torch
from .utils import get_state_dict


__all__ = ["mobilenetv2", "cifar_mobilenetv2"]

#'''
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0, dropout=0.0, **kwargs):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if input_size==32:
            s = 1
        else:
            s = 2


        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, s], # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0

        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )
        self.features = [conv_bn(3, input_channel, s)] # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t)
                    )
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t)
                    )
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        # building classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(self.last_channel, num_classes),
        )
        #self.fc = nn.Linear(self.last_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2(num_classes=1000, **kwargs):
    """
    Constructs a MobileNetV2 model.
    """
    model = MobileNetV2(num_classes=num_classes, **kwargs)
    #if config.MODEL.CHECKPOINT:
    #    state_dict = get_state_dict(config.MODEL.CHECKPOINT)
    #    model.load_state_dict(state_dict["model"])
    return model

def cifar_mobilenetv2(num_classes=1000, **kwargs):
    """
    Constructs a MobileNetV2 model.
    """
    model = MobileNetV2(num_classes=num_classes, input_size=32, **kwargs)
    #if config.MODEL.CHECKPOINT:
    #    state_dict = get_state_dict(config.MODEL.CHECKPOINT)
    #    model.load_state_dict(state_dict["model"])
    return model

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    #expand + depthwise + pointwise
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    
    def __init__(self, num_classes=10, input_size=224, **kwargs):
        super(MobileNetV2, self).__init__()

        if input_size==32:
            s = 1
        else:
            s = 2

        self.cfg = [(1,  16, 1, 1),
           (6,  24, 2, s),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def mobilenetv2(num_classes=1000, **kwargs):
    """
    Constructs a MobileNetV2 model.
    """
    model = MobileNetV2(num_classes=num_classes, **kwargs)
    #if config.MODEL.CHECKPOINT:
    #    state_dict = get_state_dict(config.MODEL.CHECKPOINT)
    #    model.load_state_dict(state_dict["model"])
    return model

def cifar_mobilenetv2(num_classes=1000, **kwargs):
    """
    Constructs a MobileNetV2 model.
    """
    model = MobileNetV2(num_classes=num_classes, input_size=32, **kwargs)
    #if config.MODEL.CHECKPOINT:
    #    state_dict = get_state_dict(config.MODEL.CHECKPOINT)
    #    model.load_state_dict(state_dict["model"])
    return model

'''