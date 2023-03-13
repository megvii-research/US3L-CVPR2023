import torch
import os
import logging
import datetime
import shlex
import subprocess
import shutil
import random

import torch.backends.cudnn as cudnn

from .ddp import is_main_process, get_rank


class EmptyClass(object):
    pass


def set_random_seed(config):
    if config.SEED is not None:
        random.seed(config.SEED)
        seed = config.SEED + get_rank()
        torch.manual_seed(seed)
        cudnn.deterministic = True


def construct_logger(name, save_dir):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    date = str(datetime.datetime.now().strftime("%m%d%H%M"))
    fh = logging.FileHandler(os.path.join(save_dir, f"log-{date}.txt"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def save_checkpoint(state, is_best, dirname, filename):
    if is_main_process():
        dst_pathname = os.path.join(dirname, filename)
        torch.save(state, dst_pathname)
        if is_best:
            shutil.copyfile(dst_pathname, os.path.join(dirname, "model_best.pth.tar"))


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt)
    return model

def load_checkpoint_wo_fc(model, path):
    ckpt = torch.load(path, map_location="cpu")
    print(model.state_dict().keys())
    print(ckpt.keys())
    print('='*10)
    for k in list(ckpt.keys()):
        if 'fc' in k:
            del ckpt[k]
    msg  = model.load_state_dict(ckpt, strict=False)
    print(msg.missing_keys)
    return model

def load_ssl_checkpoint(model, path, warmup_fc=False):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt['state_dict']
    print(state_dict.keys())
    print('='*20)
    print(model.state_dict().keys())
    print('='*20)

    if not warmup_fc:
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') and 'quantizer' not in k:
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            elif k.startswith('module.encoder') and not k.startswith('module.encoder.fc') and 'quantizer' not in k:
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            elif k.startswith('encoder_q') and not k.startswith('encoder_q.fc') and 'quantizer' not in k:
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            elif k.startswith('module') and not k.startswith('module.fc') and 'quantizer' not in k:
                state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
    
    msg = model.load_state_dict(state_dict, strict=False)
    #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print('missing', set(msg.missing_keys))
    print("=> loaded pre-trained model '{}'".format(path))
    return model

def logging_information(logger, cfg, info):
    if is_main_process():
        logger.info(info)