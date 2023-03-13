#from torchvision.models import resnet50
from thop import profile
import torch
import sys
sys.path.append('../')
from resnet_width import resnet18, resnet50
from mobilenetv2 import mobilenetv2

model = mobilenetv2(width_mult=0.25)

input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))

print('FLOPs = ' + str(flops/1000**2) + 'M')
print('Params = ' + str(params/1000**2) + 'M')