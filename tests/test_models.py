"""Tests for the three model architectures."""
from __future__ import annotations

import torch

from src.config import NUM_CLASSES, WINDOW
from src.models.lstm import LSTMModel
from src.models.mlp import MLPModel
from src.models.rnn import RNNModel


def _check_forward(model):
    x = torch.randn(8, WINDOW)
    out = model(x)
    assert out.logits.shape == (8, NUM_CLASSES)
    assert out.recon.shape == (8, WINDOW)
    return out


def test_mlp_forward():
    _check_forward(MLPModel())


def test_rnn_forward():
    _check_forward(RNNModel(hidden_size=16))


def test_lstm_forward():
    _check_forward(LSTMModel(hidden_size=16))


def test_gradient_flow():
    for model in (MLPModel(), RNNModel(hidden_size=16), LSTMModel(hidden_size=16)):
        x = torch.randn(4, WINDOW, requires_grad=False)
        out = model(x)
        loss = out.logits.sum() + out.recon.sum()
        loss.backward()
        # Every parameter should receive a gradient.
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_models_are_distinct_classes():
    a, b, c = MLPModel(), RNNModel(), LSTMModel()
    assert type(a).__name__ == "MLPModel"
    assert type(b).__name__ == "RNNModel"
    assert type(c).__name__ == "LSTMModel"
