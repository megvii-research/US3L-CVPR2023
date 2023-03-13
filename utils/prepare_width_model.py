import torch
import os
from resnet_width import resnet18, resnet50

width = 0.25

#root = '/data/train_log_OFA/imagenet/r18/simclr_momenrtum_max_with_momentum_r18_imagenet_sandwich3T_asymmertric_mse_distill_feat_2_3_4_distill_mse_groupreg_A'
#root = '/data/train_log_OFA/imagenet/r50/simclr_momenrtum_max_with_momentum_r50_imagenet_sandwich3T_asymmertric_mse_distill_feat_2_3_4_distill_mse_groupreg_A'
root = '/data/train_log_OFA/imagenet/r50/byol_us_r50_imagenet_sandwichOnce3T_asymmertric_infoncev2_distill_new_head_A_200ep'

model_path = os.path.join(root, 'checkpoint_width_{}.pth.tar'.format(width))

#root = '/data/train_log_OFA/imagenet/r18/byol_us_r18_imagenet_200ep_single_{}_rerun'.format(width)
#root = '/data/train_log_OFA/imagenet/r50/simsiam_us_r50_imagenet_100ep_single_{}'.format(width)
#model_path = os.path.join(root, 'checkpoint.pth.tar')


model = resnet50(width_mult=width)
target = model.state_dict()

src = torch.load(model_path, map_location='cpu')['state_dict']

prefix = 'module.encoder_q.'

print(target.keys())

for k, v in src.items():
    #print(k)
    if 'fc' in k or 'encoder_k' in k or 'predictor' in k:
        continue
    old_k = k
    k = k.replace(prefix, "")
    print(k, target[k].size())
    #print(target[k].size())

    if len(v.size())==1: # BN layer
        dim = target[k].size(0)
        target[k] = v[:dim]
    elif len(v.size())==4: # Conv layer
        dim0 = target[k].size(0)
        dim1 = target[k].size(1)
        #if 'conv1' in k:
        target[k] = src[old_k][:dim0,:dim1,:,:]
        #else:
        #    target[k] = src[old_k][:dim,:dim,:,:]
    else:
        print(k, target[k].size())
torch.save(target, os.path.join(root, 'checkpoint_width_{}_for_detection.pth.tar'.format(width)))

        
