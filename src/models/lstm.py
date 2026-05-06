"""LSTM model — gated memory cells that mitigate vanishing gradients."""
from __future__ import annotations

import torch
from torch import nn

from .heads import HeadOutput, MultiTaskHead


class LSTMModel(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = MultiTaskHead(hidden_size)

    def forward(self, x_noisy: torch.Tensor) -> HeadOutput:
        x = x_noisy.unsqueeze(-1)
        out, (h_n, c_n) = self.lstm(x)
        feats = h_n[-1]
        return self.head(feats)
