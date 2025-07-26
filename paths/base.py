from abc import ABC, abstractmethod
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from samplers import Sampler



class ConditionalPath(nn.Module, ABC):
    def __init__(self, noise:Sampler,init_channel:int = 3):
        super().__init__()
        self.noise = noise 
        self.init_channel = init_channel
        self.register_buffer("dummy", torch.tensor(0, ))
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z), for neural network params
        
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - xt: samples from p_t(x|z), (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_velocity(self, xt: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z) as a reference
        
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        pass


      
