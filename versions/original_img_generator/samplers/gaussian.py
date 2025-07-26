# generate guassian noise 
 
# define base class
from .base import Sampler
from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn


class IsotropicGaussian(Sampler):
    def __init__(self, sample_shape: Tuple[int], std: float = 1.0):
        super().__init__(sample_shape)
        self.register_buffer("std", torch.tensor(std))

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.sample_shape, device=self.std.device)
