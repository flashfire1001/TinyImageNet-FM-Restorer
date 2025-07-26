from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class NNVelocity(nn.Module, ABC):
    # The NN to approximate VectorField velocity
    @abstractmethod
    def forward(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - (optional) y: (bs,)
        Returns:
        - u_t^theta(x): (bs, c, h, w), or
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass
