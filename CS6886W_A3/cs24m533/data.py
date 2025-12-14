"""----------------------------------------------------------------
Modules:
    torch             : Core PyTorch library for tensor operations.
    torch.utils.data  : Utilities for dataset loading and batching.
    torchvision       : Provides datasets and computer vision utilities.
    transforms        : Defines data preprocessing and augmentation pipelines.
----------------------------------------------------------------"""
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

"""---------------------------------------------
* def name :
*       get_cifar10_dataloaders
*
* purpose:
*       Creates CIFAR-10 training and test
*       dataloaders with required transforms.
*
* Input parameters:
*       batch_size  : number of samples per batch
*       num_workers : number of data loading workers
*
* return:
*       Training and test dataloaders
---------------------------------------------"""
def get_cifar10_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
