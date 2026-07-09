# scripts/check_augment.py
import os
import torch
from models.ssl_dataset import SSLAugmentDataset, make_file_list
from models.augmentations import make_augmentations

root = "/workspace/huytq/SER/datasets/datasets"
files = make_file_list(root)
print("Found files:", len(files))
if len(files) == 0:
    raise SystemExit("No audio files found under " + root)

# show first file path
fp = files[0]
print("First file:", fp)

# create dataset (3s default)
aug = make_augmentations(sample_rate=16000)
ds = SSLAugmentDataset(files, transform=aug, sample_rate=16000, target_seconds=3.0)

# load raw waveform using dataset loader internals
wav = ds._load(fp)
wav = ds._pad_or_truncate(wav)
print("Raw waveform shape:", wav.shape, "min/max/std:",
      float(wav.min()), float(wav.max()), float(wav.std()))

# get two augmented views
v1, v2 = ds[0]
print("View1 shape/min/max/std:", v1.shape, float(v1.min()), float(v1.max()), float(v1.std()))
print("View2 shape/min/max/std:", v2.shape, float(v2.min()), float(v2.max()), float(v2.std()))

# show whether views differ from raw and from each other
diff1 = (v1 - wav).abs().mean().item()
diff2 = (v2 - wav).abs().mean().item()
diff12 = (v1 - v2).abs().mean().item()
print(f"Mean abs diff view1<->raw: {diff1:.6f}, view2<->raw: {diff2:.6f}, view1<->view2: {diff12:.6f}")