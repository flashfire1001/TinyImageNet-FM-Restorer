from .base import Simulator
from vector_fields import VectorField
import torch

class EulerSimulator(Simulator):
    def __init__(self, vector_field: VectorField):
        self.vector_field = vector_field

    @torch.no_grad()
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        return xt + self.vector_field.velocity(xt, t, **kwargs) * dt
