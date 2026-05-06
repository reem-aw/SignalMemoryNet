"""Shared multi-task output head used by all three architectures."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from ..config import OUT_LOGITS, OUT_RECON, OUT_TOTAL


class HeadOutput(NamedTuple):
    logits: torch.Tensor   # (B, OUT_LOGITS)
    recon: torch.Tensor    # (B, OUT_RECON)


class MultiTaskHead(nn.Module):
    """Linear projection that splits its output into class logits + reconstruction.

    Given a feature tensor of shape ``(B, F)`` it produces ``OUT_TOTAL`` outputs
    and slices them into the two task-specific tensors.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, OUT_TOTAL)

    def forward(self, features: torch.Tensor) -> HeadOutput:
        out = self.proj(features)
        logits = out[:, :OUT_LOGITS]
        recon = out[:, OUT_LOGITS : OUT_LOGITS + OUT_RECON]
        return HeadOutput(logits=logits, recon=recon)
