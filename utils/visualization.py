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
def visualize(x: torch.Tensor, save_path: str, dpi: int = 300):
    """
    Visualize a batch of images. Handles normalization automatically.
    Args:
        x: Tensor of shape (B, C, H, W), possibly in range [-1, 1], or unnormalized
        save_path: Where to save the resulting image
    """
    assert x.dim() == 4, f"Expected 4D tensor, got {x.shape}"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Normalize input to [0, 1]
    x = x.clone()
    if x.min() < 0:
        x = (x + 1) / 2  # Assume in [-1, 1]
    if x.max() > 1 or x.min() < 0:
        x -= x.min()
        x /= (x.max() + 1e-8)  # Min-max normalization

    # --- Construct image grid
    if x.size(0) == 1:
        img = x[0].permute(1, 2, 0).cpu().numpy()
    else:
        nrow = min(x.size(0), 8)
        grid = make_grid(x, nrow=nrow)
        img = grid.permute(1, 2, 0).cpu().numpy()

    # --- Plot and save
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

@torch.no_grad()
def generate_samples_and_save(
    model: nn.Module,
    path: ConditionalPath,
    device: torch.device,
    save_path: str,
    labels: list,
    guidance_scale: float = 5.0,
    num_steps: int = 100,
    samples_per_label: int = 1,
    simulator_type: str = "euler",
):
    model.eval()
    vector_field = CFGVectorField(model=model, guidance_scale=guidance_scale)

    if simulator_type == "euler":
        simulator = EulerSimulator(vector_field)
    elif simulator_type == "heun":
        simulator = HeunSimulator(vector_field)
    else:
        raise ValueError(f"Unsupported simulator type: {simulator_type}")

    all_images = []

    for label in labels:
        y = torch.full((samples_per_label,), label, dtype=torch.long).to(device)
        x_init = path.noise.sample(samples_per_label)
        x_init = x_init.to(device)
        ts = torch.linspace(0, 1, num_steps).view(1, -1, 1, 1, 1).expand(samples_per_label, -1, -1, -1, -1).to(device)

        x_final = simulator.simulate(x_init, ts, y=y)  # typically in [-1, 1]
        all_images.append(x_final)

    all_images = torch.cat(all_images, dim=0)  # (B, C, H, W)
    visualize(all_images, save_path)
