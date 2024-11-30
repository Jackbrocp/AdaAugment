import numpy as np
import torchvision.transforms as transforms
from Network import *
from augmentation.cutout import Cutout
from augmentation import adaaugment, trivialaugment

def make_magnitude_transform(magnitude, cutout_length):
    adaaugment.set_augmentation_space(augmentation_space='standard',num_strengths=30)
    magnitude_transform = transforms.Compose([
                transforms.ToPILImage(),
                adaaugment.AdaAugment(M=magnitude),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1,cutout_length)
            ])
    return magnitude_transform

def make_transform(dataset, length=8):
    if dataset=='cifar10' or dataset=='cifar100':
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
    return transform, transform_test
            
            