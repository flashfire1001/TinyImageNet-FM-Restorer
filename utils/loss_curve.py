import matplotlib.pyplot as plt
from typing import List
import os
from pathlib import Path

root_dir = Path("results")

def draw_loss_curve(loss_history: List[float], num_epochs: int, batch_size: int, project_name: str):
    save_path = root_dir / project_name / f"{project_name}_{num_epochs}_epochs_{batch_size}_bs_loss_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(loss_history, label=f"{project_name.upper()} Loss")
    plt.yscale('log')
    plt.title(f"{project_name.upper()} Loss Curve with {num_epochs} epochs")
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
