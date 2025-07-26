# get the dataloader with a transformed dataset.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional
def get_mnist_dataloader(
        root: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = True
) -> DataLoader:
    
    """the function to get a dataloader"""
    mean = (0.5)
    std = (0.5)

    
    train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    train_dataset = datasets.MNIST(
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
#batch_size, c, H, W
