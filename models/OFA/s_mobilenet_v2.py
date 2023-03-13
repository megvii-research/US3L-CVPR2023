import math
import torch.nn as nn


from .slimmable_ops import SwitchableBatchNorm2d, SlimmableConv2d

__all__ = ['s_mobilenetv2', 's_cifar_mobilenetv2']

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                SlimmableConv2d(inp, expand_inp, 1, 1, 0, bias=False),
                SwitchableBatchNorm2d(expand_inp),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SlimmableConv2d(
                expand_inp, expand_inp, 3, stride, 1,
                groups=expand_inp, bias=False),
            SwitchableBatchNorm2d(expand_inp),
            nn.ReLU6(inplace=True),
            SlimmableConv2d(expand_inp, outp, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, **kwargs):
        super(Model, self).__init__()

        if input_size==32:
            s = 1
        else:
            s = 2

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, s], # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        
        channel = 32
        self.outp = 1280
        first_stride = s # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    3, channel, 3,
                    first_stride, 1, bias=False, us=[False, True]),
                SwitchableBatchNorm2d(channel),
                nn.ReLU6(inplace=True))
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(channel, outp, s, t))
                else:
                    self.features.append(
                        InvertedResidual(channel, outp, 1, t))
                channel = outp

        # tail
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    channel,
                    self.outp,
                    1, 1, 0, bias=False, us=[True, False]),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        #avg_pool_size = input_size // 32
        #self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.fc = nn.Linear(self.outp, num_classes)
        #if FLAGS.reset_parameters:
        self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, self.outp)
        x = self.fc(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def s_mobilenetv2(num_classes=1000, **kwargs):
    return Model(num_classes=num_classes, **kwargs)

def s_cifar_mobilenetv2(num_classes=1000, **kwargs):
    return Model(num_classes=num_classes, input_size=32, **kwargs)