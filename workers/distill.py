import os
import random
import importlib
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import get_config, update_config
import datasets
import models
from losses import LOSS
import utils
import math
import shutil


def distill(args):
    config = get_config(args)
    config.defrost()
    device = torch.device(config.DEVICE)

    # ---- setup logger and output ----
    os.makedirs(args.output, exist_ok=True)
    logger = utils.train.construct_logger(config.SSL.TYPE, config.OUTPUT)

    if config.TRAIN.USE_DDP:  # only support single node
        utils.ddp.init_distributed_mode(config)

    cudnn.benchmark = True
    utils.train.set_random_seed(config)

    # build dataloaders
    train_preprocess  = datasets.build_ssl_transform(
        config, two_crop = True
    )

    print(train_preprocess)

    train_dataloader = datasets.build_dataloader("train", config, train_preprocess)
    
    print('build {} model'.format(config.SSL.TYPE))
    model = utils.builder.build_ssl_model(models.__dict__[config.MODEL.ARCH], config)

    if config.TRAIN.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = model.to(device)
    print(model)

    # for usnet teacher, we need to calibrate bn
    if config.MODEL.OFA.CALIBRATE_BN:
        print('-------------start bn calibration-------------')
        model.calibrate_teacher(train_dataloader)
        print('-------------finish bn calibration-------------')

    if config.TRAIN.USE_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[config.LOCAL_RANK],
            find_unused_parameters=True,
        )
    utils.logging_information(logger, config, str(model))

    criterion = LOSS(config)
    optimizer = torch.optim.SGD(model.parameters(), config.TRAIN.LR_SCHEDULER.BASE_LR,
                                momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)

    if config.MODEL.CHECKPOINT:
        ckpt = torch.load(config.MODEL.CHECKPOINT)
        config.defrost()
        config.TRAIN.START_EPOCH = ckpt['epoch']
        state_dict = ckpt['state_dict']
        print(state_dict.keys())
        print('='*20)
        print(model.state_dict().keys())
        print('='*20)
        msg = model.load_state_dict(state_dict, strict=False)
        print('state dict missing', set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(config.MODEL.CHECKPOINT))
        optimizer.load_state_dict(ckpt['optimizer'])
        print("=> loaded pre-trained optimizer")


    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if config.TRAIN.USE_DDP:
            train_dataloader.sampler.set_epoch(epoch)
        
        adjust_learning_rate(optimizer, config.TRAIN.LR_SCHEDULER.BASE_LR, epoch, config)

        train(train_dataloader, model, criterion, optimizer, epoch, config)

        if utils.is_main_process():
            utils.train.save_checkpoint(
            {
                "epoch": epoch + 1,
                'arch': config.MODEL.ARCH,
                "state_dict": model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            },
            is_best=False,
            dirname=config.OUTPUT,
            filename="checkpoint.pth.tar".format(epoch+1),
            )


def train(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    #criterion = nn.MSELoss()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        output, teacher_target = model(images[0], images[1])
        loss = criterion(output, teacher_target) 

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
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


def adjust_learning_rate(optimizer, init_lr, epoch, config):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / config.TRAIN.EPOCHS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
