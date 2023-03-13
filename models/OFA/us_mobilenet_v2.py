# !/usr/bin/env python

import torch.nn as nn
import math
import torch
from .slimmable_ops import USBatchNorm2d, USConv2d, make_divisible



__all__ = ["us_mobilenetv2", "us_cifar_mobilenetv2"]


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
    def __init__(self, inp, outp, stride, expand_ratio, bn_track_stats=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                USConv2d(
                    inp, expand_inp, 1, 1, 0, bias=False,
                    ratio=[1, expand_ratio]),
                USBatchNorm2d(expand_inp, ratio=expand_ratio, track_running_stats=bn_track_stats),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            USConv2d(
                expand_inp, expand_inp, 3, stride, 1, groups=expand_inp,
                depthwise=True, bias=False,
                ratio=[expand_ratio, expand_ratio]),
            USBatchNorm2d(expand_inp, ratio=expand_ratio, track_running_stats=bn_track_stats),

            nn.ReLU6(inplace=True),

            USConv2d(
                expand_inp, outp, 1, 1, 0, bias=False,
                ratio=[expand_ratio, 1]),
            USBatchNorm2d(outp, track_running_stats=bn_track_stats),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0, bn_track_stats=False, **kwargs):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        if input_size==32:
            s = 1
        else:
            s = 2
        
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   #features.1
            [6, 24, 2, s], # NOTE: change stride 2 -> 1 for CIFAR10, fetures.3
            [6, 32, 3, 2],  #features.6
            [6, 64, 4, 2],  #features.10
            [6, 96, 3, 1],  #features.13
            [6, 160, 3, 2], #features.16
            [6, 320, 1, 1], #features.17
        ]

        # building first layer
        assert input_size % 32 == 0

        input_channel = 32
        self.last_channel = 1280
        self.features = [nn.Sequential(
            USConv2d(
                    3, input_channel, 3, stride=s, padding=1, bias=False,  # NOTE: change conv1 stride 2 -> 1 for CIFAR10
                    us=[False, True]),
                USBatchNorm2d(input_channel, track_running_stats=bn_track_stats),
                nn.ReLU6(inplace=True)
        )]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t, bn_track_stats=bn_track_stats)
                    )
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t, bn_track_stats=bn_track_stats)
                    )
                input_channel = output_channel
        # building last several layers
        self.features.append(
            nn.Sequential(
                USConv2d(
                    input_channel, self.last_channel, 1, 1, 0, bias=False,
                    us=[True, False]),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6(inplace=True),
            )
        )

        # self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.fc = nn.Linear(self.last_channel, num_classes)
        #self.classifier = nn.Sequential(
        #    nn.Dropout(dropout),
        #    nn.Linear(self.last_channel, n_class),
        #)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
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


def us_mobilenetv2(num_classes=1000, **kwargs):
    return MobileNetV2(num_classes=num_classes, **kwargs)

def us_cifar_mobilenetv2(num_classes=1000, **kwargs):
    return MobileNetV2(num_classes=num_classes, input_size=32, **kwargs)
