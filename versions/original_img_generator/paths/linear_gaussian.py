from .gaussian import Alpha, Beta
import torch

class LinearAlpha(Alpha):
    def __init__(self):
        super().__init__()

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

class LinearBeta(Beta):
    def __init__(self):
        super().__init__()

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return - torch.ones_like(t)
