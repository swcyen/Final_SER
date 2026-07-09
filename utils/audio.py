from pathlib import Path
from typing import Union

import torch
import torchaudio


def load_waveform(
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
    duration_seconds: float = 5.0,
) -> torch.Tensor:
    """Load a .wav file as a fixed-length mono raw waveform.

    Returns:
        Tensor with shape [1, audio_samples].
    """
    waveform, original_sr = torchaudio.load(str(audio_path))  # [channels, samples]

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, samples]

    if original_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(original_sr, sample_rate)
        waveform = resampler(waveform)  # [1, resampled_samples]

    target_samples = int(sample_rate * duration_seconds)
    num_samples = waveform.size(1)

    if num_samples < target_samples:
        pad_amount = target_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))  # [1, target_samples]
    elif num_samples > target_samples:
        waveform = waveform[:, :target_samples]  # [1, target_samples]

    waveform = waveform.float()
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    return waveform
