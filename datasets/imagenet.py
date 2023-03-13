import os
import cv2
import numpy as np
import collections
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.datasets as datasets


class DATASET(datasets.ImageFolder):
    def __init__(self, transform, mode):
        if mode=='train':
            root = os.path.join('/mnt/ramdisk/ImageNet', 'train')
        else:
            root = os.path.join('/mnt/ramdisk/ImageNet', 'val')
        super(DATASET, self).__init__(root, transform)
