#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import moco.builder
import moco.loader
import moco.optimizer
from datasets.imagenet import DATASET as imagenet_dataset
#from datasets.imagenet_a import ImagenetDataset as imagenet_dataset
from datasets.cifar10 import DATASET as cifar10_dataset
import utils

#import vits
import models


torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base',  'us_vit_tiny', 'us_vit_small', 'us_vit_base', 'us_vit_conv_small', 'us_vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
parser.add_argument('--output', help='path to train log dir')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

#OFA parameters
parser.add_argument('--max-width', default=1.0, type=float, help='width multiplier')
parser.add_argument('--min-width', default=0.25, type=float, help='width multiplier')
parser.add_argument('--distill-lambda', default=1.0, type=float, help='lambda of distillation loss')
parser.add_argument('--group-regularizer', action='store_true')
parser.add_argument('--reg-lambda', default=0.05, type=float, help='reg lambda')
parser.add_argument('--reg-decay-type', default='linear', type=str, help='reg lambda')
parser.add_argument('--reg-decay-alpha', default=0.05, type=float, help='reg alpha')
parser.add_argument('--reg-decay-bins', default=8, type=int, help='reg bins')
parser.add_argument('--reg-warmup-epochs', default=100, type=int, help='reg warmup')

parser.add_argument('--baseline', action='store_true')

def main():
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    utils.init_distributed_mode(args)
    #if args.dist_url == "env://" and args.world_size == -1:
    #    args.world_size = int(os.environ["WORLD_SIZE"])

    #args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print(args.distributed)

    ngpus_per_node = torch.cuda.device_count()
    #if args.multiprocessing_distributed:
    #    # Since we have ngpus_per_node processes per node, the total world_size
    #    # needs to be adjusted accordingly
    #    args.world_size = ngpus_per_node * args.world_size
    #    # Use torch.multiprocessing.spawn to launch distributed processes: the
    #    # main_worker process function
    #    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    #else:
        # Simply call main_worker function
    #qconfig = get_config(args)
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    #if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
    #    def print_pass(*args):
    #        pass
    #    builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    #if args.distributed:
    #    utils.init_distributed_mode(args)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if 'vit' in args.arch:
        if args.baseline:
            model = moco.builder.MoCo_USBaselineViT(
                #partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
                partial(models.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
                args.moco_dim, args.moco_mlp_dim, args.moco_t)
        else:
            model = moco.builder.MoCo_USViT(
                #partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
                partial(models.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
                args.moco_dim, args.moco_mlp_dim, args.moco_t)
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay if not args.group_regularizer else 0,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay if not args.group_regularizer else 0)

    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])


    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_transform = moco.loader.TwoCropsTransform(transforms.Compose(augmentation1),
                                                    transforms.Compose(augmentation2))
    #train_dataset = imagenet_dataset(transform=train_transform, mode="train")

    train_dataset = cifar10_dataset(transform=train_transform, mode="train")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    print(args.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)

        #if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #        and args.rank == 0): # only the first GPU saves checkpoint
        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, dirname=args.output, filename='checkpoint.pth.tar')

    if args.rank == 0:
        summary_writer.close()

def generate_widths_train(args, epoch=-1):
    max_width = args.max_width
    min_width = args.min_width

     # Sample Strategy A for USNets
    #'''
    if not args.baseline:
        epinterval = args.epochs // 4
        epid = epoch // epinterval +1
        if epid == 1:
            min_width = max_width
        else:
            min_width = min_width + (4-epid)*(max_width-min_width)/(4-1)
    
    widths_train = []

    if min_width==max_width:
        widths_train = [max_width]
    # sample multiple models
    else:
        for _ in range(1):
            #widths_train.append(random.choice([0.5, 0.75]))
            widths_train.append(random.uniform(min_width, max_width))
        #if config.MODEL.OFA.SANDWICH:
        widths_train = [max_width, min_width] + widths_train
    
    widths_train.sort(reverse=True) # from max to min

    return widths_train

class MSEDistill(object):
    def __init__(self, ):
        self.name = "CosineSimilarityv2"
        self.criterion = nn.CosineSimilarity(dim=1)

    def __call__(self, pred, target):
        p1, p2 = pred
        z1,  z2 = target
        #z = (z1+z2)/2 # cal average view target
        val = (self.criterion(p1,z2.detach())+self.criterion(p2,z1.detach())).mean()
        return val

class GroupRegularizer(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.name = "GroupReg"
        self.model = model
        self.args = args
        self.reg_lambda = args.reg_lambda
        self.decay_alpha = args.reg_decay_alpha
        self.decay_type = args.reg_decay_type
        self.decay_bins = args.reg_decay_bins
        self.warmup_epochs = args.reg_warmup_epochs
        self.total_epochs = args.epochs

    def cal_lambda(self, i, epoch):
        # i: index 0,1,2,3,...,decay_bins

        if epoch<self.warmup_epochs: # normal regularizer during warmup
            return self.reg_lambda

        # dynamic
        decay_alpha = self.decay_alpha
        #ramp_epochs = 0
        ramp_epochs = self.total_epochs//2 # 200 for 400 epochs
        if epoch<ramp_epochs:
            decay_alpha *= (epoch-self.warmup_epochs) / (ramp_epochs-self.warmup_epochs)
        
        if self.decay_type == 'exp':
            return self.reg_lambda * decay_alpha**i 
        elif self.decay_type == 'linear':
            # dynamic
            return self.reg_lambda * (1-decay_alpha*i) # v1
            #return self.reg_lambda * (1-decay_alpha*(self.decay_bins-1-i)) # v2
        else:
            print('Not Implemented Error')

    def __call__(self, width_mult=1.0, epoch=-1): 
        loss = 0.
        for n, p in self.model.named_parameters():
            if 'momentum_encoder' in n or 'proj_conv' in n: # skip encoder_k
                continue 
            if width_mult==1.0 and 'predictor' in n:
                continue
            length = len(p.size())
            lambda_mask = torch.ones(p.size(), dtype=torch.float64, device=p.device)

            if length == 2: # linear, group L2
                dim_out, dim_in = p.size()[0], p.size()[1]
                group_dim_out, group_dim_in = dim_out//self.decay_bins, dim_in//self.decay_bins
                # generate mask
                for i in range(self.decay_bins):
                    end_dim = dim_in if i==self.decay_bins-1 else (i+1)*group_dim_in
                    lambda_mask[:, i*group_dim_in:end_dim] =  self.cal_lambda(i, epoch)
            else: # fc weight, normal L2
                lambda_mask[:] = self.reg_lambda
            loss += (lambda_mask * torch.square(p)).sum()

        return loss


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    reg_losses = AverageMeter('RegLoss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses, reg_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    reg_criterion = GroupRegularizer(args, model)
    distill_criterion = MSEDistill()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)

        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        widths_train = generate_widths_train(args, epoch)
        #widths_train = [1.0, 0.5]
        max_width = max(widths_train)
        for j, width_mult in enumerate(widths_train):
            #print(j, width_mult)
            model.module.base_encoder.apply(lambda m: setattr(m, 'width_mult', width_mult))
            with torch.cuda.amp.autocast(True):
            # compute output
                if width_mult == max_width:
                    loss, output, teacher_target = model(images[0], images[1], moco_m) # final version
                else:
                    loss, output, target = model(images[0], images[1], moco_m, update=False, teacher_target=teacher_target)
                    #loss = args.distill_lambda * distill_criterion(output, teacher_target)

            losses.update(loss.item(), images[0].size(0))
            #print(loss.item())
            if args.group_regularizer:
                reg_loss = reg_criterion(width_mult=width_mult, epoch=epoch)
                reg_losses.update(reg_loss.item(), images[0].size(0))
                loss += reg_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #print('finish step')
        #print('finish', i)
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def save_checkpoint(state, is_best, dirname, filename='checkpoint.pth.tar'):
    dst_name = os.path.join(dirname, filename)
    torch.save(state, dst_name)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()