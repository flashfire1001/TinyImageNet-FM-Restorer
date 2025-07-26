#composition of unet architecture
import torch
import torch.nn as nn
from ..base import NNVelocity
from .fourier_encoder import FourierEncoder
from .modules import Encoder, Midcoder, Decoder
from typing import List


class UNetVelocity(NNVelocity):
    def __init__(self, init_channel, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int, num_categories: int):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(init_channel, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.y_embedder = nn.Embedding(num_categories+1, y_embed_dim)

        encoders = []
        decoders = []
        for (curr_channel, next_channel) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_channel, next_channel, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(2*next_channel, curr_channel, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))
        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        self.final_conv = nn.Conv2d(channels[0],init_channel, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        # Embed t and y
        res = []
        t_embed = self.time_embedder(t) # (bs, t_embed_dim)
        y_embed = self.y_embedder(y) # (bs, y_embed_dim)
        
        x = self.init_conv(x) # (bs, init_channel, h, w)
        
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed) # (bs, c[i], h, w) -> (bs, c[i+1], h//2, w//2)
            res.append(x.clone()) # push res into the stack

        x = self.midcoder(x, t_embed, y_embed)

        for decoder in self.decoders:
            x_res = res.pop() # (bs, c[i], h, w) last-in, first-out
            # do concatenation , double the channels
            x = torch.cat([x, x_res], dim=1) # (bs, c[i]*2, h, w)
            x = decoder(x, t_embed, y_embed) # (bs, c[i-1], 2*h, 2*w)

        x = self.final_conv(x)

        return x
