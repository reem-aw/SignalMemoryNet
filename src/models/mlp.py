"""Fully-connected baseline (MLP) — no temporal memory."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from config.config import WINDOW
from .heads import HeadOutput, MultiTaskHead


class MLPModel(nn.Module):
    def __init__(self, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = WINDOW
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = MultiTaskHead(prev)

    def forward(self, x_noisy: torch.Tensor) -> HeadOutput:
        # x_noisy: (B, WINDOW)
        feats = self.backbone(x_noisy)
        return self.head(feats)
