"""End-to-end smoke tests for training and evaluation."""
from __future__ import annotations

import torch

from src.config import TrainConfig
from src.data import make_dataloaders
from src.evaluate import collect_predictions, confusion_matrix
from src.models.mlp import MLPModel
from src.train import evaluate_loader, train_one


def test_train_decreases_loss_on_tiny_dataset():
    cfg = TrainConfig(
        epochs=8,
        batch_size=16,
        train_records_per_freq=2,
        val_records_per_freq=1,
        test_records_per_freq=1,
        window_stride=20,
        early_stop_patience=10,
    )
    train, val, _ = make_dataloaders(
        cfg.train_records_per_freq,
        cfg.val_records_per_freq,
        cfg.test_records_per_freq,
        batch_size=cfg.batch_size,
        seed=0,
        window_stride=cfg.window_stride,
    )
    model = MLPModel()
    history = train_one(model, train, val, cfg, device=torch.device("cpu"), verbose=False)
    assert history.train_loss[-1] < history.train_loss[0]


def test_evaluate_loader_returns_three_metrics():
    cfg = TrainConfig(
        train_records_per_freq=1, val_records_per_freq=1, test_records_per_freq=1,
        window_stride=20,
    )
    _, val, _ = make_dataloaders(
        cfg.train_records_per_freq, cfg.val_records_per_freq, cfg.test_records_per_freq,
        batch_size=8, seed=0, window_stride=cfg.window_stride,
    )
    model = MLPModel()
    loss, acc, recon = evaluate_loader(model, val, torch.device("cpu"))
    assert loss > 0
    assert 0.0 <= acc <= 1.0
    assert recon >= 0


def test_confusion_matrix_shape():
    _, _, test = make_dataloaders(1, 1, 1, batch_size=8, seed=0, window_stride=20)
    model = MLPModel()
    logits, y, recon, clean, sigma = collect_predictions(model, test, torch.device("cpu"))
    cm = confusion_matrix(y, logits.argmax(dim=1), n_classes=logits.size(1))
    assert cm.shape == (4, 4)
    assert cm.sum() == y.size(0)
