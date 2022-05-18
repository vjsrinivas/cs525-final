from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from torchvision import transforms
import torchvision.datasets as datasets

# preprocess for training and testing loaders
def preprocess_train_cifar_100():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.24703233, 0.24348505, 0.26158768]),
        transforms.RandomErasing(),
    ])

def preprocess_test_cifar_100():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.24703233, 0.24348505, 0.26158768]),
    ])

def preprocess_train_cifar_10():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
        # new augs from the paper:
        transforms.RandAugment(),
        # mixup and cutmix missing
        ##
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
        transforms.RandomErasing(),
    ])

def preprocess_test_cifar_10():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])

def read_name_file(file):
    with open(file, 'r') as f:
        content = list(map(str.strip, f.readlines()))
    return content

###########################################################################
# MAKE NAMESPACES FOR EACH DATASET AND THEIR PREPROCESSING FUNCTIONS HERE:#
###########################################################################

OXFORD = SimpleNamespace(
    classes=read_name_file("data/cifar100.txt"),
)

CIFAR100 = SimpleNamespace(
    classes=read_name_file("data/cifar100.txt"),
    train_transform=preprocess_train_cifar_100,
    test_transform=preprocess_test_cifar_100
)

CIFAR10 = SimpleNamespace(
    classes=read_name_file("data/cifar10.txt"),
    train_transform=preprocess_train_cifar_10,
    test_transform=preprocess_test_cifar_10
)