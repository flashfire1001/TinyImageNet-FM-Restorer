# metadata for guassian path
from abc import ABC, abstractmethod
from .base import ConditionalPath
import torch
from torch.func import jacrev, vmap
from typing import Tuple
from samplers import IsotropicGaussian
from torch.utils.data import DataLoader

class Alpha(ABC):
   
    
    
    @abstractmethod
    def __call__(self, t:torch.Tensor)-> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        pass
       
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        da_dt = vmap(jacrev(self))(t)
        return da_dt.view(-1, 1, 1, 1).to(t.device)
    
class Beta(ABC):
        
    
    @abstractmethod
    def __call__(self, t:torch.Tensor)-> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        pass
       
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        db_dt = vmap(jacrev(self))(t)
        return db_dt.view(-1, 1, 1, 1).to(t.device)

class GaussianConditionalPath(ConditionalPath):
    def __init__(self, p_init_shape: Tuple[int], alpha: Alpha, beta: Beta):
        noise = IsotropicGaussian(p_init_shape)
        super().__init__(noise)
        self.alpha = alpha
        self.beta = beta

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_init = self.noise.sample(z.shape[0])
        xt = self.alpha(t) * z + self.beta(t) * x_init
        return xt

    def conditional_velocity(self, xt: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        d_alpha_dt = self.alpha.dt(t)
        d_beta_dt = self.beta.dt(t)
        return (d_alpha_dt - d_beta_dt * alpha_t / beta_t) * z + d_beta_dt / beta_t * xt
