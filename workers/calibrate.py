import os
import random
import importlib
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from config import get_config, update_config
import datasets
import models
from losses import LOSS
import utils
import math
import shutil
import losses
from utils.forward_hook import ForwardHookManager
from copy import deepcopy
from datasets.custom_transforms import GaussianBlur, TwoCropsTransform

class MODEL(nn.Module):
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(MODEL, self).__init__()

        self.config = config
        
        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM
        self.m = config.SSL.SETTING.MOMENTUM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=hidden_dim, bn_track_stats=True)
        
        print(self.encoder_q.state_dict().keys())
        print(self.encoder_q)
        
    def forward(self, x):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """

        #online_proj_one = self.encoder_q(x1)
        #online_proj_two = self.encoder_q(x2)
        #return [online_proj_one, online_proj_two]
        return self.encoder_q(x)

def calibrate(args):
    config = get_config(args)
    config.defrost()
    device = torch.device(config.DEVICE)

    # ---- setup logger and output ----
    os.makedirs(args.output, exist_ok=True)
    logger = utils.train.construct_logger(config.SSL.TYPE, config.OUTPUT)

    if config.TRAIN.USE_DDP:  # only support single node
        utils.ddp.init_distributed_mode(config)

    cudnn.benchmark = True
    #utils.train.set_random_seed(config)

    # build dataloaders
    #train_preprocess  = datasets.build_ssl_transform(
    #    config, two_crop = True
    #)
    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    print(train_preprocess)

    train_dataloader = datasets.build_dataloader("train", config, train_preprocess)
    
    print('build {} model'.format(config.SSL.TYPE))
    model = MODEL(models.__dict__[config.MODEL.ARCH], config)

    if config.MODEL.PRETRAINED:
        ckpt = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
        state_dict = ckpt['state_dict']
        print(state_dict.keys())
        print('='*20)
        print(model.state_dict().keys())
        print('='*20)
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.', '')] = state_dict[k]
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print('state dict missing', set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(config.MODEL.PRETRAINED))

    model = model.to(device)
    print(model)

    if config.TRAIN.USE_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[config.LOCAL_RANK],
            find_unused_parameters=True,
        )
    utils.logging_information(logger, config, str(model))

    model.apply(lambda m: setattr(m, 'width_mult', config.MODEL.OFA.WIDTH_MULT))
    model.train()
    for i, (images, target) in enumerate(train_dataloader):
        #if args.gpu is not None:
        print(i, images[0].size())
        #images[0] = images[0].cuda(non_blocking=True)
        #images[1] = images[1].cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            model(images)
        if i==500: # partial images enough
            break
    if utils.is_main_process():
        utils.train.save_checkpoint(
        {
            'arch': config.MODEL.ARCH,
            "state_dict": model.state_dict(),
        },
        is_best=False,
        dirname=config.OUTPUT,
        filename="checkpoint_width_{}.pth.tar".format(config.MODEL.OFA.WIDTH_MULT),
        )
