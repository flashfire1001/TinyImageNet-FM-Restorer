# get the dataloader with a transformed dataset.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional
def get_cifar10_dataloader(
        root: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = True
) -> DataLoader:
    
    """the function to get a dataloader"""
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.CIFAR10(
        root = root,
        train = True,
        download = download,
        transform = train_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,)
    
    return train_dataloader
