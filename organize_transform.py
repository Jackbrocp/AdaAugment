import pathlib
import sys
# sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
import argparse
import os
from typing import Generator
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Network import *
from augmentation.cutout import Cutout
from augmentation import gridmask,fast_autoaugment, HaS, Grid, RandomErasing
from augmentation import CIFAR10Policy, ImageNetPolicy, trivialaugment, CIFAR10Policy_magnitude, AdaRandAugment

def make_magnitude_transform_TA(magnitude, cutout_length):
    # TAugment = trivialaugment.AdaTrivialAugment(M=magnitude)
    trivialaugment.set_augmentation_space(augmentation_space='standard',num_strengths=30)
    magnitude_transform = transforms.Compose([
                transforms.ToPILImage(),
                trivialaugment.AdaTrivialAugment(M=magnitude),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1,cutout_length)
            ])
    return magnitude_transform

def make_magnitude_transform_AA(magnitude, cutout_length):
    magnitude_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                CIFAR10Policy_magnitude(M = magnitude),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1, length=cutout_length)
            ])
    return magnitude_transform

def make_magnitude_transform_RA(magnitude, cutout_length):
    magnitude_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                AdaRandAugment(1,magnitude),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1, length=cutout_length)
            ])
    return magnitude_transform

def make_transform(dataset, aug, length=8, M=6):
    if dataset=='cifar10' or dataset=='cifar100':
        if aug == 'autoaugment':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1, length)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
        elif aug == 'autoaugment_magnitude':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy_magnitude(M = M),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1, length)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
        elif aug == 'randaugment':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                RandAugment(1,30),
                # RandAugment(3,M),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1, length)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
        elif aug == 'has':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                HaS()
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
        elif aug == 'cutout':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1, length)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
        elif aug == 'trivialaugment':
            TAugment = trivialaugment.TrivialAugment()
            trivialaugment.set_augmentation_space(augmentation_space='standard',num_strengths=30)
            transform = transforms.Compose([
                        transforms.ToPILImage(),
                        TAugment,
                        transforms.RandomCrop(32,padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                        Cutout(1,length)
                    ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                        ])
        elif aug == 'gridmask':
            transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomCrop(32,padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                        Grid(d1=24, d2=33, rotate=1, ratio=0.4, mode=1, prob=0.8)
                    ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                        ])
        elif aug == 'fast-autoaugment':
            transform, transform_test = fast_autoaugment.get_transform(dataset)
        elif aug == 'randomerasing':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                RandomErasing( ),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
            
    elif dataset=='tiny-imagenet':
        if aug == 'autoaugment':
            transform=transforms.Compose([
                #  transforms.RandomResizedCrop(64), 
                 transforms.RandomCrop(64, padding=4),
                 transforms.RandomHorizontalFlip(), 
                 ImageNetPolicy(), 
                 transforms.ToTensor(), 
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif aug == 'fast-autoaugment':
            transform, transform_test = fast_autoaugment.get_transform(dataset)
    
    return transform, transform_test
            
            