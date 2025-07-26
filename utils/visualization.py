import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pathlib import Path
from vector_fields import VectorField, CFGVectorField
from simulators import EulerSimulator, HeunSimulator
from paths import ConditionalPath


@torch.no_grad()
def generate_samples_and_save(
    model: nn.Module,
    path: ConditionalPath,
    device: torch.device,
    file_name:str, 
    project_name:str,
    labels: list,
    guidance_scales:list[float],
    num_steps: int = 100,
    samples_per_label: int = 1,
    simulator_type: str = "euler",
    dpi:int = 300
):
    #initialize the model, vector field and the simulator
    model.eval()
    
    
    # depict the cifar10 to generate
    fig, axes = plt.subplots(len(guidance_scales), 1, figsize=(10, 5 * len(guidance_scales)))
    for idx, w in enumerate(guidance_scales):
        vector_field = CFGVectorField(model=model, guidance_scale=w)
        if simulator_type == "euler":
            simulator = EulerSimulator(vector_field)
        elif simulator_type == "heun":
            simulator = HeunSimulator(vector_field)
        else:
            raise ValueError(f"Unsupported simulator type: {simulator_type}")

        #y: all labels a axis need.
        y = torch.tensor(labels, dtype=torch.int64).repeat_interleave(samples_per_label).to(device) #(#samples_per_label*len(labels),)
        x0 = path.noise.sample(y.shape[0]) # sample a collection of noises
        ts = torch.linspace(0, 1, num_steps).view(1, -1, 1, 1, 1).expand(y.shape[0], -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=y)
        # reshape x1 to get images in a correct form.
        images = x1
        images = images.reshape(len(labels), samples_per_label, 3, 32, 32)
        images = images.permute((1, 0, 2, 3, 4))
        images = images.reshape(-1, 3, 32, 32)
        print("simulation and generation finished, start display and render...")
        print("Any NaNs:", torch.isnan(images).any().item())
        print("Any Infs:", torch.isinf(images).any().item())
        print("Min:", images.min().item(), "Max:", images.max().item())
          
        grid = make_grid(images.cpu(), nrow=len(labels), normalize=True, value_range=(0, 1))
        axes[idx].imshow(grid.permute(1, 2, 0).cpu())
        axes[idx].axis("off")
        axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=20)

    # --- Plot and save
    save_path = "results" / Path(project_name) /f"{file_name}_guidances{guidance_scales}.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.show()
    
    