"""Training loop shared by all three architectures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import LAMBDA_CLS, LAMBDA_RECON, TrainConfig


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_recon_mse: List[float] = field(default_factory=list)


def _step(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device):
    x = batch["x_noisy"].to(device)
    y_class = batch["y_class"].to(device)
    y_clean = batch["x_clean"].to(device)
    out = model(x)
    cls_loss = nn.functional.cross_entropy(out.logits, y_class)
    rec_loss = nn.functional.mse_loss(out.recon, y_clean)
    loss = LAMBDA_CLS * cls_loss + LAMBDA_RECON * rec_loss
    return loss, out, y_class, y_clean


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_recon = 0.0
    n = 0
    for batch in loader:
        loss, out, y_class, y_clean = _step(model, batch, device)
        bs = y_class.size(0)
        total_loss += loss.item() * bs
        total_correct += (out.logits.argmax(dim=1) == y_class).sum().item()
        total_recon += nn.functional.mse_loss(out.recon, y_clean, reduction="sum").item()
        n += bs
    return total_loss / n, total_correct / n, total_recon / (n * out.recon.size(1))


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device | None = None,
    verbose: bool = True,
) -> History:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = History()
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            opt.zero_grad()
            loss, *_ = _step(model, batch, device)
            loss.backward()
            opt.step()
            bs = batch["x_noisy"].size(0)
            running += loss.item() * bs
            n += bs
        train_loss = running / max(1, n)
        val_loss, val_acc, val_recon = evaluate_loader(model, val_loader, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.val_recon_mse.append(val_recon)
        if verbose:
            print(
                f"epoch {epoch:3d} | train {train_loss:.4f} "
                f"| val {val_loss:.4f} | acc {val_acc:.3f} | recon {val_recon:.4f}"
            )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                if verbose:
                    print(f"early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
