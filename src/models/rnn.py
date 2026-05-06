"""Vanilla RNN model — each sample is a timestep, hidden state is the memory."""
from __future__ import annotations

import torch
from torch import nn

from .heads import HeadOutput, MultiTaskHead


class RNNModel(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.head = MultiTaskHead(hidden_size)

    def forward(self, x_noisy: torch.Tensor) -> HeadOutput:
        # x_noisy: (B, WINDOW) -> reshape to (B, T=WINDOW, F=1)
        x = x_noisy.unsqueeze(-1)
        out, h_n = self.rnn(x)
        # Use the final hidden state of the last layer.
        feats = h_n[-1]
        return self.head(feats)
