## the model: predict the average vectorfield formed by guassian paths

# the model is backboned with unet, which is composed of Encoder, decoder, midcoder and skip connection.
# residual layers are emphasized and fourier encoder is adopted for time perception


from .fm_unet import UNetVelocity
from .base import NNVelocity

__all__ = ["NNVelocity","UNetVelocity"]

