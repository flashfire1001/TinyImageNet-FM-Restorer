import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
from utils import model_size_mib, save_checkpoint, draw_loss_curve

class Trainer(ABC):
    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def get_train_loss(self, data, device) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_scaler(self):
        device_name = 'cuda' if torch.cuda.is_available else 'cpu'
        return torch.amp.GradScaler(device_name)

    def get_grad_clip_norm(self) -> float:
        return 1.0

    def get_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=1000,
            min_lr=1e-6
        )

    def train(self, num_epochs: int, device: torch.device,
              dataloader: torch.utils.data.DataLoader, project_name: str, lr: float=1e-3):
        # print the model size
        model_size = model_size_mib(self.model)
        print(f'Training model with size: {model_size} MiB')

        # train start
        self.model.to(device)
        self.model.train()

        #trainer modules setup
        optimizer = self.get_optimizer(lr)
        scheduler = self.get_scheduler(optimizer)
        scaler = self.get_scaler()
        clip_norm = self.get_grad_clip_norm()

        loss_history = []
        pbar = tqdm(enumerate(range(num_epochs)))

        # train loop
        for index, epoch in pbar:
            for batch_idx, data in enumerate(dataloader):
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
                    loss = self.get_train_loss(data=data, device=device)

                # Mixed precision backward + gradient clip
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()

                scheduler.step(loss.item())  # default: ReduceLROnPlateau
                loss_history.append(loss.item())
                pbar.set_description(f"Epoch {index} batch {batch_idx}, Loss: {loss.item():.4f}")

        # train finish
        self.model.eval()
        draw_loss_curve(loss_history, num_epochs=num_epochs,
                        batch_size=dataloader.batch_size, project_name=project_name)
        save_checkpoint(self.model, optimizer, scheduler, scaler,
                        epoch=num_epochs, loss_history=loss_history,
                        project_name=project_name, filename=f"{project_name}_model.pth")
