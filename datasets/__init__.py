import importlib
from PIL import Image
import torch
from torchvision import transforms

from utils import logging_information
from utils import ddp as ddp_utils
from config import update_config

_INTERPOLATION_MODE_MAPPING = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "box": Image.BOX,
    "hamming": Image.HAMMING,
    "lanczos": Image.LANCZOS,
}

def build_ssl_transform(cfg, two_crop=True):
    from datasets.custom_transforms import GaussianBlur, TwoCropsTransform
    if 'cifar' in cfg.TRAIN.DATASET:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        print('cifar ssl transform')
        transform_list = [transforms.RandomResizedCrop(cfg.MODEL.INPUTSHAPE, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5), # remove gaussianblur for cifar
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]
    else:
        print('imagenet ssl transform')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_list = [transforms.RandomResizedCrop(cfg.MODEL.INPUTSHAPE, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]
    if two_crop:
        return TwoCropsTransform(transforms.Compose(transform_list))
    else:
        return transforms.Compose(transform_list)

def build_eval_transform(cfg):
    transform_list = []
    need_resized_input = True  # a flag specified add a resize transform in ending

    if cfg.AUG.EVALUATION.RESIZE.ENABLE:
        _cfg = cfg.AUG.EVALUATION.RESIZE
        transform_list.append(
            transforms.Resize(
                size=workaround_torch_size_bug(_cfg.SIZE)
                if _cfg.KEEP_RATIO
                else _cfg.SIZE,
                interpolation=_INTERPOLATION_MODE_MAPPING[_cfg.INTERPOLATION],
            )
        )
        need_resized_input = False

    if cfg.AUG.EVALUATION.CENTERCROP.ENABLE:
        transform_list.append(
            transforms.CenterCrop(workaround_torch_size_bug(cfg.MODEL.INPUTSHAPE))
        )
        need_resized_input = False

    if need_resized_input:
        transform_list.append(
            transforms.Resize(workaround_torch_size_bug(cfg.MODEL.INPUTSHAPE))
        )

    transform_list.append(transforms.ToTensor())

    if cfg.AUG.EVALUATION.NORMLIZATION.MEAN:
        transform_list.append(
            transforms.Normalize(
                mean=cfg.AUG.EVALUATION.NORMLIZATION.MEAN, std=[1, 1, 1]
            )
        )
    if cfg.AUG.EVALUATION.NORMLIZATION.STD:
        transform_list.append(
            transforms.Normalize(
                mean=[0, 0, 0],
                std=cfg.AUG.EVALUATION.NORMLIZATION.STD,
            )
        )

    return transforms.Compose(transform_list)


def build_dataloader(mode, config, preprocess):
    if mode == "calibration":
        if config.QUANT.CALIBRATION.TYPE == "tar":
            dataset = importlib.import_module("datasets.tardata").DATASET(
                config.QUANT.CALIBRATION.PATH, preprocess
            )
            if config.QUANT.CALIBRATION.SIZE > len(dataset):
                update_config(config, "QUANT.CALIBRATION.SIZE", len(dataset))
        elif config.QUANT.CALIBRATION.TYPE == "python_module":
            raise NotImplementedError("TODO")
        else:
            raise NotImplementedError(
                "No support {}".format(config.QUANT.CALIBRATION.TYPE)
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.QUANT.CALIBRATION.BATCHSIZE,
            shuffle=False,
            num_workers=config.QUANT.CALIBRATION.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        dataset_class = importlib.import_module(
            "datasets." + config.TRAIN.DATASET
        ).DATASET
        dataset = dataset_class(transform=preprocess, mode=mode)
        if mode == "train":
            if config.TRAIN.USE_DDP:
                num_tasks = ddp_utils.get_world_size()
                global_rank = ddp_utils.get_rank()
                train_sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                train_sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=config.TRAIN.BATCH_SIZE,  # per-gpu
                num_workers=config.TRAIN.NUM_WORKERS,
                pin_memory=True,
                drop_last = config.TRAIN.DROP_LAST,
                shuffle=True if train_sampler is None else False,
            )
        elif mode == "validation":
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=min(256, config.TRAIN.BATCH_SIZE),
                #batch_size = 32,
                num_workers=config.TRAIN.NUM_WORKERS,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
    return dataloader
