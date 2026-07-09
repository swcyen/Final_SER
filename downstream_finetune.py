import argparse
import yaml
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset import load_visec_datasets
from models.encoder import SpeechEncoder


def probe_cuda_available() -> bool:
    """Return True if CUDA appears available and nvidia-smi responds.

    This is a best-effort probe; absence of nvidia-smi does not strictly
    mean CUDA is unavailable, so we also check torch.cuda.is_available().
    """
    try:
        import subprocess
        # Try nvidia-smi; if it's not present this will raise or return non-zero
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            return torch.cuda.is_available()
    except Exception:
        return torch.cuda.is_available()
    return torch.cuda.is_available()


def build_classifier(in_dim: int, num_classes: int, dropout: float = 0.5):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_dim, num_classes)
    )


def load_ssl_encoder_from_checkpoint(encoder: nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model_state", state)
    # model_state may contain keys like 'encoder.xxx' if saved from SSLModel
    filtered = {}
    for k, v in model_state.items():
        if k.startswith("encoder."):
            newk = k.replace("encoder.", "")
            filtered[newk] = v
        elif k.startswith("encoder_"):
            # some checkpoints may use underscore
            newk = k.replace("encoder_", "")
            filtered[newk] = v
    if len(filtered) == 0:
        # maybe full encoder saved directly
        try:
            encoder.load_state_dict(model_state, strict=False)
            return
        except Exception:
            raise RuntimeError("Checkpoint does not contain encoder weights with expected keys.")

    encoder.load_state_dict(filtered, strict=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--mode", choices=["linear_probe", "finetune"], default="linear_probe")
    p.add_argument("--ssl-checkpoint", type=str, default=None, help="Path to SSL checkpoint to initialize encoder")
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    p.add_argument("--lr", type=float, default=None, help="Learning rate for the optimizer")
    p.add_argument("--device", type=str, default=("cuda" if probe_cuda_available() else "cpu"), help="Device to use for training")
    p.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    p.add_argument("--no-fallback", action="store_true", help="If set, do not fallback to CPU when CUDA/NVML fails")
    return p.parse_args()


def train_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device, criterion, scaler: Optional[torch.amp.GradScaler] = None, log_interval: int = 20):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    running = 0.0
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad()
        with torch.amp.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optim.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        running += loss.item()
        if step % log_interval == 0:
            avg = running / log_interval
            print(f"Step {step} AvgLoss {avg:.4f}")
            running = 0.0

    return total_loss / total, correct / total


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_ds, val_ds, test_ds, class_names = load_visec_datasets(cfg)
    num_classes = len(class_names)

    # Determine device with optional fallback if CUDA/NVML not available
    requested = args.device
    cuda_ok = (requested == "cuda") and probe_cuda_available() and torch.cuda.is_available()
    if requested == "cuda" and not cuda_ok:
        msg = "Requested CUDA but CUDA/NVML not available; falling back to CPU."
        if args.no_fallback:
            raise RuntimeError(msg + " Set --no-fallback to prevent fallback.")
        else:
            print(msg)
            device = torch.device("cpu")
    else:
        device = torch.device(requested)

    model_cfg = cfg.get("model", {})
    encoder = SpeechEncoder(
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
        feedforward_dim=model_cfg.get("feedforward_dim", 512),
        dropout=model_cfg.get("dropout", 0.1),
        pooling=model_cfg.get("pooling", "mean"),
    )

    # load ssl checkpoint if provided
    if args.ssl_checkpoint:
        load_ssl_encoder_from_checkpoint(encoder, args.ssl_checkpoint, device)

    # classifier
    classifier = build_classifier(encoder.embedding_dim, num_classes, dropout=model_cfg.get("classifier_dropout", 0.5))

    # combine into a single nn.Module for convenience
    class DownstreamModel(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, x):
            emb = self.encoder(x)
            return self.classifier(emb)

    model = DownstreamModel(encoder, classifier).to(device)

    # freeze encoder if linear_probe
    if args.mode == "linear_probe":
        for p in model.encoder.parameters():
            p.requires_grad = False

    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 16)
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 20)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 1e-4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.get("data", {}).get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    params = filter(lambda p: p.requires_grad, model.parameters())
    # default small weight decay for stability if not specified in config
    default_wd = cfg.get("training", {}).get("weight_decay", None)
    if default_wd is None:
        default_wd = 1e-5
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=default_wd)
    criterion = nn.CrossEntropyLoss()
    use_amp = cfg.get("training", {}).get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    best_val_acc = 0.0
    best_path = os.path.join(args.save_dir, "best_downstream.pt")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion, scaler)
        val_loss, val_acc = eval_epoch(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer": optimizer.state_dict()}, best_path)
            print(f"Saved best model to {best_path} (val_acc={best_val_acc:.4f})")

    # final test using best
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model_state"], strict=False)
    test_loss, test_acc = eval_epoch(model, test_loader, device, criterion)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
