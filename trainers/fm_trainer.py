import torch
import torch.nn as nn
from .base import Trainer
from paths import ConditionalPath
from models import NNVelocity

class CFGTrainerFM(Trainer):
    def __init__(self, path: ConditionalPath, model: NNVelocity, eta: float):
        assert (eta <= 1) and (eta >= 0)
        super().__init__(model)
        self.path = path
        self.eta = eta

    def get_train_loss(self, data, device) -> torch.Tensor:
        z, y = data
        z = z.to(device) #data
        y = y.to(device) #labels
        # mask the image label with rate eta
        mask = torch.rand(z.shape[0]).to(device)
        y[mask < self.eta] = 10.0

        t = torch.rand(z.shape[0], 1, 1, 1).to(device)
        xt = self.path.sample_conditional_path(z, t)

        u_t_ref = self.path.conditional_velocity(xt, z, t)      # shape(bs, c, h, w)
        u_t_theta = self.model(xt, t, y)                              # shape(bs, c, h, w)

        # error = torch.einsum("bchw -> b", torch.square(u_t_ref - u_t_theta))    # shape(bs, )
        # return torch.mean(error)
        error = torch.mean(torch.square(u_t_ref - u_t_theta),dim = 0).sum() #should be of shape()
        return error
