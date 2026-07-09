import random
import math
import torch
import torch.nn.functional as F
from typing import Optional


class AudioAugmentations:
    """Simple, dependency-light audio augmentations returning torch.Tensor.

    Works with mono audio. Inputs may be 1D (samples,) or 2D (1, samples).
    Outputs are `torch.FloatTensor` shaped `(1, samples)` suitable for the
    `SpeechEncoder` in this repo (expects `[batch, 1, samples]`).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_factor: float = 0.005,
        shift_max: float = 0.2,
        gain_db: float = 6.0,
        reverb_prob: float = 0.25,
        specaug_prob: float = 0.2,
        max_crop_pct: float = 0.15,
        max_time_mask_pct: float = 0.1,
        clip_prob: float = 0.2,
    ):
        self.sample_rate = sample_rate
        self.noise_factor = noise_factor
        self.shift_max = shift_max
        self.gain_db = gain_db
        self.reverb_prob = reverb_prob
        self.specaug_prob = specaug_prob
        self.max_crop_pct = max_crop_pct
        self.max_time_mask_pct = max_time_mask_pct
        self.clip_prob = clip_prob

    def _to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        # make shape (1, samples)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tensor(x)
        std = x.std() if x.numel() > 0 else 1.0
        noise = torch.randn_like(x) * (self.noise_factor * std)
        return x + noise

    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tensor(x)
        samples = x.shape[-1]
        max_shift = int(self.shift_max * samples)
        if max_shift <= 0:
            return x
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(x, shifts=shift, dims=-1)

    def random_gain(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tensor(x)
        db = random.uniform(-self.gain_db, self.gain_db)
        gain = 10 ** (db / 20.0)
        return x * gain

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tensor(x)
        maxv = x.abs().max()
        if maxv > 0:
            return x / maxv
        return x

    def random_crop_or_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly crop up to `max_crop_pct` fraction of the audio and pad back to original length."""
        x = self._to_tensor(x)
        n = x.shape[-1]
        if self.max_crop_pct <= 0:
            return x
        crop_pct = random.uniform(0.0, self.max_crop_pct)
        keep = int(n * (1.0 - crop_pct))
        if keep >= n or keep <= 0:
            return x
        start = random.randint(0, n - keep)
        cropped = x[..., start : start + keep]
        # pad back to n
        pad_left = random.randint(0, n - keep)
        pad_right = n - keep - pad_left
        return F.pad(cropped, (pad_left, pad_right))

    def time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out a contiguous time span up to `max_time_mask_pct` of the signal."""
        x = self._to_tensor(x)
        n = x.shape[-1]
        if self.max_time_mask_pct <= 0:
            return x
        mask_pct = random.uniform(0.0, self.max_time_mask_pct)
        mask_len = int(n * mask_pct)
        if mask_len <= 0:
            return x
        start = random.randint(0, max(0, n - mask_len))
        x[..., start : start + mask_len] = 0.0
        return x

    def random_clip(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tensor(x)
        if random.random() < self.clip_prob:
            thr = random.uniform(0.3, 0.95)
            return x.clamp(min=-thr, max=thr)
        return x

    def reverb(self, x: torch.Tensor, min_rt60: float = 0.02, max_rt60: float = 0.18) -> torch.Tensor:
        """Apply a lightweight exponential-decay impulse response via conv1d.

        This is dependency-light: we generate a short IR and convolve with it.
        """
        x = self._to_tensor(x)
        if random.random() >= self.reverb_prob:
            return x
        n = x.shape[-1]
        ir_len = random.randint(int(min_rt60 * self.sample_rate), max(int(max_rt60 * self.sample_rate), 1))
        times = torch.arange(ir_len, dtype=torch.float32) / float(self.sample_rate)
        # random decay rate
        rt60 = random.uniform(min_rt60, max_rt60)
        # create exponential decay and slight randomization
        decay = torch.exp(-times * (3.0 / rt60))
        noise = 0.6 + 0.4 * torch.rand_like(decay)
        ir = decay * noise
        ir = ir / (ir.sum() + 1e-9)

        # conv1d expects shape (B, C, L)
        inp = x.unsqueeze(0)  # (1, 1, L)
        weight = ir.view(1, 1, -1)
        # pad so output length matches input
        padded = F.pad(inp, (ir_len - 1, 0))
        out = F.conv1d(padded, weight)
        out = out.squeeze(0)
        return out

    def spec_augment(self, x: torch.Tensor, n_fft: int = 512, hop_length: Optional[int] = None, freq_mask_max: int = 10, time_mask_max: int = 10) -> torch.Tensor:
        """Apply simple SpecAugment (time & frequency masking) using STFT/ISTFT.

        This uses torch.stft/istft and is reasonably fast on CPU.
        """
        x = self._to_tensor(x)
        if random.random() >= self.specaug_prob:
            return x
        if hop_length is None:
            hop_length = n_fft // 4
        # remove leading channel
        wav = x.squeeze(0)
        L = wav.shape[-1]
        # pad to at least n_fft
        if L < n_fft:
            wav = F.pad(wav, (0, n_fft - L))
        # use a Hann window for STFT/ISTFT to reduce spectral leakage and
        # ensure the same window is used for inversion (suppresses warnings)
        window = torch.hann_window(n_fft, device=wav.device)
        st = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = st.abs()
        # frequency mask
        f = random.randint(0, min(freq_mask_max, mag.shape[0] - 1))
        f0 = random.randint(0, max(0, mag.shape[0] - f)) if f > 0 else 0
        if f > 0:
            mag[f0 : f0 + f, :] = 0
        # time mask
        t = random.randint(0, min(time_mask_max, mag.shape[1] - 1))
        t0 = random.randint(0, max(0, mag.shape[1] - t)) if t > 0 else 0
        if t > 0:
            mag[:, t0 : t0 + t] = 0
        # reconstruct using original phase
        st_aug = mag * torch.exp(1j * torch.angle(st))
        wav_rec = torch.istft(st_aug, n_fft=n_fft, hop_length=hop_length, window=window, length=L)
        return wav_rec.unsqueeze(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random chain of augmentations and return a tensor shaped (1, samples)."""
        x = self._to_tensor(x)
        # core augmentations
        if random.random() < 0.9:
            x = self.add_noise(x)
        if random.random() < 0.8:
            x = self.time_shift(x)
        # slight random crop/pad to simulate different segment lengths
        if random.random() < 0.6:
            x = self.random_crop_or_pad(x)
        # optional spec-augment (STFT domain masks)
        if random.random() < self.specaug_prob:
            try:
                x = self.spec_augment(x)
            except Exception:
                # spec augment may fail for very short signals; ignore
                pass
        # occasionally add reverb
        if random.random() < self.reverb_prob:
            x = self.reverb(x)
        # random gain and clipping
        if random.random() < 0.6:
            x = self.random_gain(x)
        x = self.random_clip(x)
        # time mask (zero out short spans)
        if random.random() < 0.4:
            x = self.time_mask(x)
        x = self.normalize(x)
        return x


def make_augmentations(sample_rate: int = 16000) -> AudioAugmentations:
    return AudioAugmentations(sample_rate=sample_rate)
