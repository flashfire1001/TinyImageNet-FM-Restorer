from abc import ABC, abstractmethod
import torch

class VectorField(ABC):
    @abstractmethod
    def velocity(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1)
        Returns:
            - velocity: (bs, c, h, w)
        """
        pass
