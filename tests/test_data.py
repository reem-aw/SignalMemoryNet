"""Tests for the synthetic-signal data pipeline."""
from __future__ import annotations

import numpy as np
import torch

from src.config import NUM_CLASSES, SAMPLES_PER_RECORD, WINDOW
from src.data import SignalWindowDataset, generate_signal, make_dataloaders


def test_generate_signal_shapes_and_alignment():
    rng = np.random.default_rng(0)
    clean, noisy = generate_signal(freq=5.0, amplitude=1.0, phase=0.0,
                                   sigma_pct=0.1, rng=rng)
    assert clean.shape == (SAMPLES_PER_RECORD,)
    assert noisy.shape == (SAMPLES_PER_RECORD,)
    # Noisy = clean + noise, so the residual std should match sigma_pct * A.
    residual = noisy - clean
    assert abs(residual.std() - 0.1) < 0.02


def test_generate_signal_zero_noise_is_clean():
    clean, noisy = generate_signal(freq=2.0, amplitude=1.0, phase=0.5,
                                   sigma_pct=0.0,
                                   rng=np.random.default_rng(1))
    assert np.allclose(clean, noisy)


def test_generate_signal_deterministic_with_seed():
    a_clean, a_noisy = generate_signal(10.0, 1.0, 0.0, 0.1,
                                       rng=np.random.default_rng(123))
    b_clean, b_noisy = generate_signal(10.0, 1.0, 0.0, 0.1,
                                       rng=np.random.default_rng(123))
    assert np.array_equal(a_clean, b_clean)
    assert np.array_equal(a_noisy, b_noisy)


def test_dataset_item_structure():
    ds = SignalWindowDataset(records_per_freq=2, seed=7, stride=10)
    item = ds[0]
    assert item["x_noisy"].shape == (WINDOW,)
    assert item["x_clean"].shape == (WINDOW,)
    assert item["y_onehot"].shape == (NUM_CLASSES,)
    assert item["y_class"].dtype == torch.long
    assert torch.isclose(item["y_onehot"].sum(), torch.tensor(1.0))
    # Length matches records * windows-per-record.
    assert len(ds) == 2 * NUM_CLASSES * ds._n_starts


def test_dataset_class_balance():
    ds = SignalWindowDataset(records_per_freq=4, seed=3, stride=20)
    counts = [0] * NUM_CLASSES
    for i in range(len(ds)):
        counts[int(ds[i]["y_class"])] += 1
    assert all(c == counts[0] for c in counts)


def test_make_dataloaders_runs():
    train, val, test = make_dataloaders(2, 1, 1, batch_size=8, seed=0, window_stride=20)
    batch = next(iter(train))
    assert batch["x_noisy"].shape[1] == WINDOW
    assert batch["y_class"].dtype == torch.long
    # Val/test loaders are non-empty.
    assert len(val.dataset) > 0
    assert len(test.dataset) > 0
