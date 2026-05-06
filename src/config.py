"""Project-wide configuration constants for the SignalMemoryNet HW1.

Centralising these values keeps the data pipeline, models, training loop and
report generator in agreement.  Choices are explained in the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# Four chosen frequencies (Hz). With fs = 100 Hz the highest (20 Hz) sits well
# above the Nyquist limit of 50 Hz, so reconstruction is information-theoretically
# possible (lecture: "to recreate a sine signal we need to sample twice the
# frequency per second").
FREQUENCIES_HZ: Tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
NUM_CLASSES: int = len(FREQUENCIES_HZ)

# Sampling.
SAMPLE_RATE_HZ: float = 100.0
RECORD_SECONDS: float = 10.0
SAMPLES_PER_RECORD: int = int(SAMPLE_RATE_HZ * RECORD_SECONDS)  # 1000

# Sliding context window length (matches HW spec item 6).
WINDOW: int = 10

# Noise levels (sigma is a percentage of the amplitude A).
NOISE_LEVELS: Tuple[float, ...] = (0.05, 0.10, 0.20)

# Amplitude jitter range.
AMPLITUDE_RANGE: Tuple[float, float] = (0.8, 1.2)

# Output split: 4 logits for the 1-hot frequency + WINDOW clean reconstruction.
OUT_LOGITS: int = NUM_CLASSES
OUT_RECON: int = WINDOW
OUT_TOTAL: int = OUT_LOGITS + OUT_RECON  # 14

# Loss weights.
LAMBDA_CLS: float = 1.0
LAMBDA_RECON: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42
    train_records_per_freq: int = 30
    val_records_per_freq: int = 8
    test_records_per_freq: int = 8
    window_stride: int = 5
    hidden_size: int = 32
    mlp_hidden: Tuple[int, int] = field(default_factory=lambda: (64, 64))
    early_stop_patience: int = 5
