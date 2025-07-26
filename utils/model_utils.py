import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional


MiB = 1024 ** 2

#set the root directory for saving checkpoints
model_root_dir = Path("checkpoints")


def model_size_mib(model: nn.Module) -> float:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size / MiB


def save_checkpoint(model, optimizer, scheduler, scaler, epoch,batch_size:int, loss_history,
                    project_name: str, filename: str, final:bool = False):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "batch_size":batch_size ,
        "loss_history": loss_history if final else None
    }

    # optimizer and scheduler are optionally saved.
    # 
    save_path = model_root_dir / project_name / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] saved to {save_path}")

    # if is_best:
    #     best_path = save_path.parent / "best.pth"
    #     torch.save(checkpoint, best_path)
    #     print(f"[Checkpoint] Also saved as best model to {best_path}")


def load_checkpoint(model, optimizer, scheduler, scaler,
                    project_name: str, filename: str, device: torch.device = None)->Tuple[int, int, Optional[list]]:
    
    #use filename because we have both final and midst checkpoints
    path = model_root_dir / project_name / filename
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and checkpoint.get(checkpoint["optimizer_state_dict"]) is not None :
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    batch_size = checkpoint["batch_size"]
    loss_history = checkpoint.get("loss_history", None)

    if loss_history:
        print(f"[Checkpoint] Loaded from {path}, epoch = {epoch}, loss = {loss_history[-1]:.4f}")
    else:
        print(f"[Checkpoint] Loaded from {path}, epoch = {epoch}, loss = N/A")

    return epoch, batch_size, loss_history
