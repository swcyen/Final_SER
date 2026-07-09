# Vietnamese Speech Emotion Recognition Baseline

Lightweight end-to-end SER baseline in PyTorch. The model learns directly from
raw `.wav` waveform audio and does not use MFCC or Mel-Spectrogram input
features.

## Dataset

The training pipeline uses the Hugging Face ViSEC dataset directly:

```text
hustep-lab/ViSEC
```

Dataset link:
https://huggingface.co/datasets/hustep-lab/ViSEC

ViSEC is loaded with `datasets.load_dataset`, then split deterministically into
train/validation/test subsets according to `configs/config.yaml`.

## Setup

```bash
pip install -r requirements.txt
```

## Train

Edit `configs/config.yaml` if needed, then run:

```bash
python train.py
```

The script prints the class mapping, device, and trainable parameter count.
TensorBoard logs are written to `runs/baseline_ser`, and the best validation
checkpoint is saved to `checkpoints/best_model.pt`.

```bash
tensorboard --logdir runs
```

## Evaluate

```bash
python evaluate.py
```

This evaluates the saved best checkpoint on `dataset/test`, prints accuracy,
weighted F1, and a classification report, then saves:

```text
outputs/confusion_matrix.png
outputs/classification_report.txt
```

## Inference

```bash
python inference.py --audio path/to/audio.wav
```

The inference pipeline loads audio with `torchaudio`, resamples to 16 kHz,
converts to mono, normalizes, and pads or truncates to 5 seconds.
