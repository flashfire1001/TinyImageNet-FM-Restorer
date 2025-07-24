import torch
import torch.nn as nn
from pathlib import Path

MiB = 1024 ** 2
root_dir = Path("results")


def model_size_mib(model: nn.Module) -> float:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size / MiB


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss_history,
                    project_name: str, filename: str, is_best: bool = False):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "loss_history": loss_history
    }

    save_path = root_dir / project_name / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved to {save_path}")

    if is_best:
        best_path = save_path.parent / "best.pth"
        torch.save(checkpoint, best_path)
        print(f"[Checkpoint] Also saved as best model to {best_path}")


def load_checkpoint(model, optimizer, scheduler, scaler,
                    project_name: str, filename: str, device: torch.device = None):
    path = root_dir / project_name / filename
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    loss_history = checkpoint.get("loss_history", None)

    if loss_history:
        print(f"[Checkpoint] Loaded from {path}, epoch = {epoch}, loss = {loss_history[-1]:.4f}")
    else:
        print(f"[Checkpoint] Loaded from {path}, epoch = {epoch}, loss = N/A")

    return epoch, loss_history
