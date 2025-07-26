# do a quick test of capability of the trained generative model

import torch
from datasets import get_mnist_dataloader,get_cifar10_dataloader
from paths import GaussianConditionalPath, LinearAlpha, LinearBeta
from models import UNetVelocity
from trainers import CFGTrainerFM
from utils import generate_samples_and_save,load_checkpoint
import matplotlib.pyplot as plt
from vector_fields import CFGVectorField
from simulators import EulerSimulator
from torchvision.utils import make_grid
from matplotlib.axes._axes import Axes
from pathlib import Path


#set the root directory for saving outputs
project_name = "fm_unet_cifar10"
print(f"do the project:{project_name}")
model_path = Path("results")/project_name
image_path = Path("checkpoints") / project_name
model_path.mkdir(parents=True, exist_ok = True)
image_path.mkdir(parents = True, exist_ok = True)


#device agnostic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")

dataloader = get_cifar10_dataloader()
#print("dataloader is done")
init_channel = next(iter(dataloader))[0].shape[1]
noise_shape = next(iter(dataloader))[0].shape[1:]

path = GaussianConditionalPath(
    p_init_shape = noise_shape,
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

model_for_test = UNetVelocity(
    init_channel = init_channel,
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 64,
    y_embed_dim = 64,
    num_categories = 10
).to(device)


foo, bar,_ = load_checkpoint(model=model_for_test,
                optimizer = None,
                scheduler = None,
                scaler = None,
                project_name = project_name,
                filename = f"{project_name}_epoch80_bs128_model.pth",
                device = device)

generate_samples_and_save(
    model = model_for_test,
    path = path,
    device = device,
    file_name = "testimage_1",
    project_name= project_name,
    labels = [i for i in range(1, 11)],
    guidance_scales=[2.0,3.0,5.0],
    samples_per_label = 3, # new "batch size" for the generation process
    simulator_type = "heun"
)