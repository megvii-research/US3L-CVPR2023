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
import losses
from utils.forward_hook import ForwardHookManager
from copy import deepcopy

from torch.cuda.amp import autocast as autocast, GradScaler


def ssl(args):
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
    train_preprocess = datasets.build_ssl_transform(
        config, two_crop=True
    )

    print(train_preprocess)

    train_dataloader = datasets.build_dataloader("train", config, train_preprocess)

    print('build {} model'.format(config.SSL.TYPE))
    model = utils.builder.build_ssl_model(models.__dict__[config.MODEL.ARCH], config)

    if config.MODEL.PRETRAINED:
        ckpt = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
        state_dict = ckpt['state_dict']
        print(state_dict.keys())
        print('=' * 20)
        print(model.state_dict().keys())
        print('=' * 20)
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.', '')] = state_dict[k]
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print('state dict missing', set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(config.MODEL.PRETRAINED))

    if config.MODEL.CHECKPOINT:
        ckpt = torch.load(config.MODEL.CHECKPOINT, map_location='cpu')
        config.defrost()
        config.TRAIN.START_EPOCH = ckpt['epoch']
        state_dict = ckpt['state_dict']
        print(state_dict.keys())
        print('=' * 20)
        print(model.state_dict().keys())
        print('=' * 20)
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.', '')] = state_dict[k]
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print('state dict missing', set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(config.MODEL.CHECKPOINT))
        # optimizer.load_state_dict(ckpt['optimizer'])

    if config.TRAIN.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)
    print(model)

    if config.TRAIN.USE_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[config.LOCAL_RANK],
            find_unused_parameters=True,
        )
    utils.logging_information(logger, config, str(model))

    # hack: set weight decay to zero if we use custom regularizer
    if config.MODEL.OFA.REGULARIZER:
        update_config(config, 'TRAIN.OPTIMIZER.WEIGHT_DECAY', 0.0)

    criterion = LOSS(config)
    optimizer = torch.optim.SGD(model.parameters(), config.TRAIN.LR_SCHEDULER.BASE_LR,
                                momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    if config.MODEL.CHECKPOINT:
        # optimizer.load_state_dict(ckpt['optimizer'])
        print("=> loaded pre-trained optimizer")

    scaler = GradScaler(enabled=config.TRAIN.USE_AMP)
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if config.TRAIN.USE_DDP:
            train_dataloader.sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, config.TRAIN.LR_SCHEDULER.BASE_LR, epoch, config)

        train(train_dataloader, model, criterion, optimizer, scaler, epoch, config, logger)

        if utils.is_main_process():
            utils.train.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    'arch': config.MODEL.ARCH,
                    "state_dict": model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                is_best=False,
                dirname=config.OUTPUT,
                filename="checkpoint.pth.tar".format(epoch + 1),
            )
            # save for further evaluation on ImageNet
            if 'imagenet' in config.TRAIN.DATASET and (epoch + 1) % 50 == 0:
                shutil.copyfile(os.path.join(config.OUTPUT, 'checkpoint.pth.tar'),
                                os.path.join(config.OUTPUT, 'checkpoint_{:04d}.pth.tar'.format(epoch)))


def generate_widths_train(config, epoch=-1):
    max_width = config.MODEL.OFA.WIDTH_MULT_RANGE[1]
    min_width = config.MODEL.OFA.WIDTH_MULT_RANGE[0]

    # Sample Strategy A for USNets
    # '''
    if config.MODEL.OFA.SAMPLE_SCHEDULER == "A":
        epinterval = config.TRAIN.EPOCHS // 4
        epid = epoch // epinterval + 1
        if epid == 1:
            min_width = max_width
        else:
            min_width = min_width + (4 - epid) * (max_width - min_width) / (4 - 1)


    widths_train = []
    if config.MODEL.OFA.USE_SLIMMABLE:
        widths_train = config.MODEL.OFA.WIDTH_MULT_LIST
    else:
        # use single model
        if config.MODEL.OFA.NUM_SAMPLE_TRAINING == 1:
            widths_train = [config.MODEL.OFA.WIDTH_MULT]
        # use maximum model
        elif min_width == max_width:
            widths_train = [max_width]
        # sample multiple models
        else:
            if config.MODEL.OFA.SANDWICH:
                for _ in range(config.MODEL.OFA.NUM_SAMPLE_TRAINING - 2):
                    widths_train.append(
                        random.uniform(min_width, max_width))
                    widths_train = [max_width, min_width] + widths_train
            else:
                for _ in range(config.MODEL.OFA.NUM_SAMPLE_TRAINING):
                    widths_train.append(random.uniform(min_width, max_width))
    widths_train.sort(reverse=True)  # from max to min

    return widths_train


def train(train_loader, model, criterion, optimizer, scaler, epoch, config, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    reg_losses = AverageMeter('RegLoss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, reg_losses],
        prefix="Epoch: [{}]".format(epoch))

    distill_criterion = importlib.import_module(
        "losses.criterions." + config.MODEL.OFA.DISTILL_CRITERION).Criterion(config)
    if config.MODEL.OFA.DISTILL_FEATURE:
        feat_distill_criterion = importlib.import_module(
            "losses.criterions." + config.MODEL.OFA.DISTILL_FEATURE_CRITERION).Criterion(config)
        feat_distill_criterion_mse = importlib.import_module(
            "losses.criterions." + 'MSELoss').Criterion(config)

    if config.MODEL.OFA.REGULARIZER:
        reg_criterion = importlib.import_module(
            "losses.regularizers." + config.MODEL.OFA.REGULARIZER_CRITERION).Regularizer(config, model)

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        widths_train = generate_widths_train(config, epoch)
        # hack: use maximum width in the sample list, instead of the original width range[1]
        max_width = max(widths_train)

        # teacher_target_list = []

        # optimizer.zero_grad()
        for j, width_mult in enumerate(widths_train):
            # print(j, width_mult)

            if config.MODEL.OFA.DISTILL_FEATURE:  # if use intermediate feature
                model.module.proj_conv.apply(lambda m: setattr(m, 'width_mult', width_mult))
            if config.MODEL.OFA.DISTILL:  # if we use distill, we keep encoder_k max width (1.0)
                model.module.encoder_q.apply(lambda m: setattr(m, 'width_mult', width_mult))
            else:  # if we don't use distill, we also set encoder_k to width_mult
                model.apply(lambda m: setattr(m, 'width_mult', width_mult))

            # if we use ensemble teacher, we need to set encoder_k to different width
            # model.module.encoder_k.apply(lambda m: setattr(m, 'width_mult', width_mult))
            with autocast(enabled=config.TRAIN.USE_AMP):
                if width_mult == max_width:  # teacher branch
                    # output, teacher_target = model(images[0], images[1])
                    if config.MODEL.OFA.DISTILL_FEATURE:
                        output, teacher_target, teacher_feat_q, teacher_feat_k = model(images[0], images[1],
                                                                                       ret_q_feature=False,
                                                                                       ret_k_feature=True,
                                                                                       proj_conv=False)
                    else:
                        # output, self_target, teacher_target = model(images[0], images[1]) # hack : when max model without momentum
                        output, teacher_target = model(images[0], images[1])  # final version
                    # loss = criterion(output, self_target) # hack : when max model without momentum
                    loss = criterion(output, teacher_target)  # final version
                else:  # student branch
                    # whether to update momentum encoder in subnetwork's forward pass
                    momentum_update = config.MODEL.OFA.MOMENTUM_UPDATE
                    if config.MODEL.OFA.DISTILL_FEATURE:
                        output, target, student_feat_q, _ = model(images[0], images[1], ret_q_feature=True,
                                                                  ret_k_feature=False, proj_conv=True,
                                                                  momentum_update=momentum_update)
                    else:
                        output, target = model(images[0], images[1], momentum_update=momentum_update)

                    if config.MODEL.OFA.DISTILL:
                        # use max_width teacher model
                        # loss = criterion(output, teacher_target)
                        loss = config.MODEL.OFA.DISTILL_LAMBDA * distill_criterion(output, teacher_target, epoch)

                        # if use intermediate feature
                        if config.MODEL.OFA.DISTILL_FEATURE:
                            feat_distill_loss = feat_distill_criterion(student_feat_q, teacher_feat_k, epoch)
                            loss += config.MODEL.OFA.DISTILL_FEATURE_LAMBDA * feat_distill_loss
                    else:
                        loss = criterion(output, target)  # use self target

            losses.update(loss.item(), images[0].size(0))

            if config.MODEL.OFA.REGULARIZER:
                reg_loss = reg_criterion(width_mult=width_mult, epoch=epoch)
                reg_losses.update(reg_loss.item(), images[0].size(0))
                loss += reg_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            # normal SGD
            # loss.backward()
            # optimizer.step()
            # mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
            common_info = "Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t".format(
                epoch,
                i,
                len(train_loader),
                batch_time=batch_time,
            )
            utils.logging_information(logger, config, common_info + str(criterion))


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


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return 0.5 * (1. + math.cos(math.pi * current / rampdown_length))


def adjust_learning_rate(optimizer, init_lr, epoch, config):
    warmup_epochs = config.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS
    if epoch < warmup_epochs:
        cur_lr = linear_rampup(epoch, warmup_epochs) * init_lr
    else:
        """Decay the learning rate based on schedule"""
        # cur_lr = init_lr * cosine_rampdown(epoch-warmup_epochs, config.TRAIN.EPOCHS-warmup_epochs)
        if config.TRAIN.LR_SCHEDULER.TYPE == 'cosine':
            cur_lr = init_lr * cosine_rampdown(epoch - warmup_epochs, config.TRAIN.EPOCHS - warmup_epochs)
        else:
            cur_lr = init_lr
            for milestone in config.TRAIN.LR_SCHEDULER.DECAY_MILESTONES:
                cur_lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res