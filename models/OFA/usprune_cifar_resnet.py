'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .slimmable_ops import PrunedConv2d, PrunedBatchNorm2d

from torch.autograd import Variable
__all__ = ["usprune_cifar_resnet18",  "usprune_cifar_resnet34", "usprune_cifar_resnet50", "usprune_cifar_resnet101", "usprune_cifar_resnet152"]

class CIFAR_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_track_stats = False):
        super(CIFAR_BasicBlock, self).__init__()
        self.conv1 = PrunedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = PrunedBatchNorm2d(planes, track_running_stats=bn_track_stats)
        self.conv2 = PrunedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = PrunedBatchNorm2d(planes, track_running_stats=bn_track_stats)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                PrunedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                PrunedBatchNorm2d(self.expansion*planes, track_running_stats=bn_track_stats)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CIFAR_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_track_stats = False):
        super(CIFAR_Bottleneck, self).__init__()
        self.conv1 = PrunedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = PrunedBatchNorm2d(planes, track_running_stats=bn_track_stats)
        self.conv2 = PrunedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = PrunedBatchNorm2d(planes, track_running_stats=bn_track_stats)
        self.conv3 = PrunedConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = PrunedBatchNorm2d(self.expansion*planes, track_running_stats=bn_track_stats)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                PrunedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                PrunedBatchNorm2d(self.expansion*planes, track_running_stats=bn_track_stats)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=128, zero_init_residual=True, bn_track_stats = False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.bn_track_stats = bn_track_stats

        self.conv1 = PrunedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = PrunedBatchNorm2d(64,  track_running_stats=self.bn_track_stats)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        # self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.bn_track_stats))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = self.l2norm(out)
        return out

'''
from models.resnet import resnet18
def cifar_resnet18(num_classes=128, **kwargs):
    model = resnet18(num_classes=128, **kwargs)
    model.conv1 = PrunedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
#'''

def usprune_cifar_resnet18(num_classes=128, **kwargs):
    return ResNet(CIFAR_BasicBlock, [2,2,2,2], num_classes, **kwargs)

def usprune_cifar_resnet34(num_classes=128, **kwargs):
    return ResNet(CIFAR_BasicBlock, [3,4,6,3], num_classes, **kwargs)

def usprune_cifar_resnet50(num_classes=128, **kwargs):
    return ResNet(CIFAR_Bottleneck, [3,4,6,3], num_classes, **kwargs)

def usprune_cifar_resnet101(num_classes=128, **kwargs):
    return ResNet(CIFAR_Bottleneck, [3,4,23,3], num_classes, **kwargs)

def usprune_cifar_resnet152(num_classes=128, **kwargs):
    return ResNet(CIFAR_Bottleneck, [3,8,36,3], num_classes, **kwargs)


def test():
    net = usprune_cifar_resnet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()