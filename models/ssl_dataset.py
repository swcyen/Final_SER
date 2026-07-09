import os
from typing import List, Optional
import torch
from torch.utils.data import Dataset

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False

from .augmentations import AudioAugmentations


class SSLAugmentDataset(Dataset):
    """Dataset that loads audio files and returns two augmented views.

    Each item is a tuple `(view1, view2)` where both are tensors shaped
    `(1, samples)` (mono). The dataset will pad or truncate audio to
    `target_seconds` * `sample_rate`.
    """

    def __init__(self, filepaths: List[str], transform: Optional[AudioAugmentations] = None, sample_rate: int = 16000, target_seconds: float = 3.0):
        self.filepaths = list(filepaths)
        self.transform = transform or AudioAugmentations(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * target_seconds)

    def __len__(self):
        return len(self.filepaths)

    def _load(self, path: str) -> torch.Tensor:
        if _HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(path)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            # ensure shape (1, samples)
            return waveform.float()
        else:
            # fallback: use soundfile if present, otherwise raise
            try:
                import soundfile as sf
            except Exception:
                raise RuntimeError("Neither torchaudio nor soundfile are available to load audio files.")
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            import numpy as np
            if sr != self.sample_rate:
                try:
                    import resampy
                    data = resampy.resample(data, sr, self.sample_rate)
                except Exception:
                    raise RuntimeError("Resampling required but `resampy` is not installed.")
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            return tensor

    def _pad_or_truncate(self, wav: torch.Tensor) -> torch.Tensor:
        # wav shape (1, samples)
        cur = wav.shape[-1]
        if cur == self.target_samples:
            return wav
        elif cur > self.target_samples:
            start = (cur - self.target_samples) // 2
            return wav[..., start:start + self.target_samples]
        else:
            # pad
            pad_amount = self.target_samples - cur
            left = pad_amount // 2
            right = pad_amount - left
            return torch.nn.functional.pad(wav, (left, right))

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        wav = self._load(path)
        wav = self._pad_or_truncate(wav)
        # produce two augmented views
        view1 = self.transform(wav.clone())
        view2 = self.transform(wav.clone())
        return view1, view2


def make_file_list(root: str, extensions=None):
    if extensions is None:
        extensions = {".wav", ".flac", ".mp3"}
    out = []
    for dp, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in extensions:
                out.append(os.path.join(dp, f))
    return sorted(out)
