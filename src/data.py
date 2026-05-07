"""Synthetic sine-wave dataset for HW1.

Each record is a 10-second signal of one of the four chosen frequencies.  The
signal is generated *twice*: once clean (target) and once noisy (input).  We
then slide a length-`WINDOW` context across the record to produce supervised
samples whose targets are (1) the 1-hot frequency code, (2) the matching clean
window for reconstruction, and (3) the noise level (returned for analysis).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config.config import (
    AMPLITUDE_RANGE,
    FREQUENCIES_HZ,
    NOISE_LEVELS,
    NUM_CLASSES,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_RECORD,
    WINDOW,
)


def generate_signal(
    freq: float,
    amplitude: float,
    phase: float,
    sigma_pct: float,
    n_samples: int = SAMPLES_PER_RECORD,
    fs: float = SAMPLE_RATE_HZ,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (clean, noisy) signal arrays of shape ``(n_samples,)``.

    Implements ``(A ± sigma)·sin(2π f t + φ)`` plus additive Gaussian noise of
    standard deviation ``sigma_pct * A`` on the noisy copy.
      The clean copy is
    free of additive noise but uses the same ``A`` and ``φ`` so the two copies
    are aligned sample-by-sample (HW spec item 3).
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n_samples, dtype=np.float64) / fs
    clean = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    noise = rng.normal(loc=0.0, scale=sigma_pct * amplitude, size=n_samples)
    noisy = clean + noise
    return clean.astype(np.float32), noisy.astype(np.float32)


@dataclass
class _Record:
    freq_idx: int
    sigma_pct: float
    clean: np.ndarray
    noisy: np.ndarray


class SignalWindowDataset(Dataset):
    """Sliding-window dataset over many synthetic sine records.

    Item layout (all tensors)::

        x_noisy : float32, shape (WINDOW,)
        x_clean : float32, shape (WINDOW,)
        y_class : long,   scalar (frequency index in [0, NUM_CLASSES))
        y_onehot: float32, shape (NUM_CLASSES,)
        sigma   : float32, scalar (noise level used for this record)
    """

    def __init__(
        self,
        records_per_freq: int,
        seed: int = 0,
        window: int = WINDOW,
        stride: int = 5,
    ) -> None:
        self.window = window
        self.stride = stride
        rng = np.random.default_rng(seed)
        self._records: List[_Record] = []
        for freq_idx, freq in enumerate(FREQUENCIES_HZ):
            for _ in range(records_per_freq):
                A = float(rng.uniform(*AMPLITUDE_RANGE))
                phi = float(rng.uniform(0.0, 2.0 * np.pi))
                sigma_pct = float(rng.choice(NOISE_LEVELS))
                clean, noisy = generate_signal(freq, A, phi, sigma_pct, rng=rng)
                self._records.append(_Record(freq_idx, sigma_pct, clean, noisy))

        # Pre-compute window starting offsets per record.
        n_starts = max(1, (SAMPLES_PER_RECORD - window) // stride + 1)
        self._n_starts = n_starts

    def __len__(self) -> int:
        return len(self._records) * self._n_starts

    def __getitem__(self, idx: int):
        rec_idx, win_idx = divmod(idx, self._n_starts)
        rec = self._records[rec_idx]
        start = win_idx * self.stride
        end = start + self.window
        x_noisy = torch.from_numpy(rec.noisy[start:end].copy())
        x_clean = torch.from_numpy(rec.clean[start:end].copy())
        y_class = torch.tensor(rec.freq_idx, dtype=torch.long)
        y_onehot = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        y_onehot[rec.freq_idx] = 1.0
        sigma = torch.tensor(rec.sigma_pct, dtype=torch.float32)
        return {
            "x_noisy": x_noisy,
            "x_clean": x_clean,
            "y_class": y_class,
            "y_onehot": y_onehot,
            "sigma": sigma,
        }


def make_dataloaders(
    train_records_per_freq: int,
    val_records_per_freq: int,
    test_records_per_freq: int,
    batch_size: int = 128,
    seed: int = 42,
    window_stride: int = 5,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SignalWindowDataset(train_records_per_freq, seed=seed, stride=window_stride)
    val_ds = SignalWindowDataset(val_records_per_freq, seed=seed + 1, stride=window_stride)
    test_ds = SignalWindowDataset(test_records_per_freq, seed=seed + 2, stride=window_stride)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
