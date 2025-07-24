from .base import Simulator
from vector_fields import VectorField
import torch


class HeunSimulator(Simulator):
    def __init__(self, vector_field: VectorField):
        self.vector_field = vector_field


    @torch.no_grad()
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        vt = self.vector_field.velocity(xt, t, **kwargs)
        xt_next = xt + vt * dt
        t_next = t + dt
        vt_next = self.vector_field.velocity(xt_next, t_next, **kwargs)
        return xt + 0.5 * dt * (vt + vt_next)
