from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        class_names: List[str],
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.class_names = class_names

        train_cfg = config["training"]
        path_cfg = config["paths"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        self.use_amp = bool(train_cfg.get("mixed_precision", True) and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.epochs = train_cfg["epochs"]
        self.early_stopping_patience = train_cfg["early_stopping_patience"]

        self.best_checkpoint = Path(path_cfg["best_checkpoint"])
        self.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=path_cfg["log_dir"])

    def fit(self) -> None:
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self._train_one_epoch(epoch)
            val_loss, val_metrics = self.evaluate(self.val_loader)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("accuracy/val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("weighted_f1/train", train_metrics["weighted_f1"], epoch)
            self.writer.add_scalar("weighted_f1/val", val_metrics["weighted_f1"], epoch)
            self.writer.add_scalar("lr", current_lr, epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['weighted_f1']:.4f} lr={current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, best_val_loss, val_metrics)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        self.writer.close()

    def _train_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        all_labels: List[int] = []
        all_preds: List[int] = []

        progress = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        for waveform, labels in progress:
            waveform = waveform.to(self.device, non_blocking=True)  # [batch, 1, samples]
            labels = labels.to(self.device, non_blocking=True)  # [batch]

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(waveform)  # [batch, num_classes]
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * waveform.size(0)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss, compute_metrics(all_labels, all_preds)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_labels: List[int] = []
        all_preds: List[int] = []

        for waveform, labels in tqdm(data_loader, desc="Eval", leave=False):
            waveform = waveform.to(self.device, non_blocking=True)  # [batch, 1, samples]
            labels = labels.to(self.device, non_blocking=True)  # [batch]

            logits = self.model(waveform)  # [batch, num_classes]
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * waveform.size(0)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

        avg_loss = total_loss / len(data_loader.dataset)
        return avg_loss, compute_metrics(all_labels, all_preds)

    def _save_checkpoint(self, epoch: int, val_loss: float, val_metrics: Dict[str, float]) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "class_names": self.class_names,
            "config": self.config,
        }
        torch.save(checkpoint, self.best_checkpoint)
        print(f"Saved best checkpoint: {self.best_checkpoint}")
