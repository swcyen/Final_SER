import os
import sys
import argparse
import math
import time
import torch
from torch.utils.data import DataLoader

from models.ssl_dataset import SSLAugmentDataset, make_file_list
from models.augmentations import make_augmentations
from models.nt_xent import NTXentLoss
from models.encoder import SpeechEncoder
from models.ssl_model import SSLModel


def probe_cuda_available() -> bool:
    # Require both `torch.cuda.is_available()` and a working `nvidia-smi` tool.
    try:
        import subprocess
        if not torch.cuda.is_available():
            return False
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False


def collate_fn(batch):
    # batch: list of (view1, view2), each (1, samples)
    v1 = torch.stack([b[0] for b in batch], dim=0)
    v2 = torch.stack([b[1] for b in batch], dim=0)
    return v1, v2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/workspace/huytq/SER/datasets/datasets", help="Path to dataset root containing audio files")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g. 'cpu' or 'cuda'). If omitted the script will probe for CUDA at runtime.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup steps for LR scaling")
    p.add_argument("--pin-memory", action="store_true", help="Use pin_memory in DataLoader")
    p.add_argument("--no-fallback", action="store_true", help="If set and CUDA requested but unavailable, exit with error instead of falling back to CPU")
    p.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")
    args = p.parse_args()
    if args.device is None:
        args.device = ("cuda" if probe_cuda_available() else "cpu")
    return args


def set_lr_with_schedule(optimizer, base_lr, step, total_steps, warmup_steps):
    # linear warmup then cosine decay multiplicative factor
    if step < warmup_steps:
        lr_mult = float(step) / float(max(1, warmup_steps))
    else:
        # cosine decay from 1 -> 0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    lr = base_lr * lr_mult
    for g in optimizer.param_groups:
        g["lr"] = lr


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    files = make_file_list(args.data)
    if len(files) == 0:
        raise RuntimeError(f"No audio files found under {args.data}. Provide a path with .wav/.flac/.mp3 files.")

    # Device handling: decide early so we can set DataLoader pin_memory correctly.
    requested = args.device
    cuda_ok = ("cuda" in str(requested)) and probe_cuda_available()
    if "cuda" in str(requested) and not cuda_ok:
        if args.no_fallback:
            print("Requested CUDA but CUDA/NVML not available; exiting due to --no-fallback.")
            sys.exit(1)
        print("Requested CUDA but CUDA/NVML not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        try:
            device = torch.device(requested)
        except Exception:
            device = torch.device("cpu")

    aug = make_augmentations(sample_rate=args.sample_rate)
    dataset = SSLAugmentDataset(files, transform=aug, sample_rate=args.sample_rate, target_seconds=args.seconds)

    # If GPU is used, enable pin_memory by default unless user explicitly disabled it.
    pin_memory_flag = args.pin_memory or (device.type == "cuda")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=pin_memory_flag)

    encoder = SpeechEncoder()
    model = SSLModel(encoder)

    model.to(device)

    # Use AdamW with small weight decay to improve stability and avoid collapse
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = NTXentLoss(temperature=args.temperature)

    total_steps = args.epochs * math.ceil(len(dataset) / args.batch_size)
    global_step = 0

    use_amp = torch.cuda.is_available() and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    start_epoch = 1
    if args.resume is not None:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck.get("model_state", ck), strict=False)
        optimizer.load_state_dict(ck.get("optimizer", optimizer.state_dict()))
        start_epoch = ck.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        epoch_start = time.time()
        for step, (v1, v2) in enumerate(loader, start=1):
            v1 = v1.to(device)
            v2 = v2.to(device)

            global_step += 1
            set_lr_with_schedule(optimizer, args.lr, global_step, total_steps, args.warmup_steps)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=(scaler is not None)):
                z1 = model(v1)
                z2 = model(v2)
                loss = loss_fn(z1, z2)
            #     pos_sim = (
            #         z1 * z2
            #     ).sum(dim=1).mean()

            # print(f"Positive similarity: {pos_sim.item()}")
            # std = z1.std(dim=0).mean()

            # print(f"Standard deviation: {std.item()}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running += loss.item()
            if step % 20 == 0:
                avg = running / 20
                print(f"Epoch {epoch} Step {step} AvgLoss {avg:.4f} LR {optimizer.param_groups[0]['lr']:.6g}")
                running = 0.0

        epoch_time = time.time() - epoch_start

        # Diagnostics: sample a few batches and compute embedding statistics
        def compute_diagnostics(model, loader, device, max_batches=8):
            model.eval()
            zs1 = []
            zs2 = []
            with torch.no_grad():
                for i, (a, b) in enumerate(loader):
                    a = a.to(device)
                    b = b.to(device)
                    z1 = model(a)
                    z2 = model(b)
                    zs1.append(z1.detach().cpu())
                    zs2.append(z2.detach().cpu())
                    if i + 1 >= max_batches:
                        break
            if len(zs1) == 0:
                return {}
            z1 = torch.cat(zs1, dim=0)
            z2 = torch.cat(zs2, dim=0)
            z = torch.cat([z1, z2], dim=0)
            per_dim_std = z.std(dim=0)
            mean_std = per_dim_std.mean().item()
            pos_sim = (z1 * z2).sum(dim=1).mean().item()
            sim = z @ z.t()
            n = sim.shape[0]
            with_neg = sim.sum().item() - sim.diagonal().sum().item()
            neg_count = n * n - n
            neg_mean = with_neg / max(1, neg_count)
            z_centered = z - z.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(z_centered)
                top_s = s[:10].cpu().tolist()
            except Exception:
                top_s = []
            return {"mean_std": mean_std, "pos_sim": pos_sim, "neg_mean": neg_mean, "top_svals": top_s}

        stats = compute_diagnostics(model, loader, device)
        ckpt_path = os.path.join(args.save_dir, f"ssl_epoch{epoch}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
        print(f"Epoch {epoch} done in {epoch_time:.1f}s, saved checkpoint: {ckpt_path}")
        if stats:
            print(f"Diag: mean_std={stats['mean_std']:.4f} pos_sim={stats['pos_sim']:.4f} neg_mean={stats['neg_mean']:.4f} top_svals={stats['top_svals']}")

    print("Training finished")


if __name__ == "__main__":
    main()
