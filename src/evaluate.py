"""Evaluation + plotting helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config.config import FREQUENCIES_HZ, NOISE_LEVELS, NUM_CLASSES
from .train import History


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_y, all_recon, all_clean, all_sigma = [], [], [], [], []
    for batch in loader:
        x = batch["x_noisy"].to(device)
        out = model(x)
        all_logits.append(out.logits.cpu())
        all_recon.append(out.recon.cpu())
        all_y.append(batch["y_class"])
        all_clean.append(batch["x_clean"])
        all_sigma.append(batch["sigma"])
    return (
        torch.cat(all_logits),
        torch.cat(all_y),
        torch.cat(all_recon),
        torch.cat(all_clean),
        torch.cat(all_sigma),
    )


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def plot_history(histories: Dict[str, History], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, h in histories.items():
        axes[0].plot(h.train_loss, label=f"{name} train")
        axes[0].plot(h.val_loss, label=f"{name} val", linestyle="--")
        axes[1].plot(h.val_acc, label=name)
        axes[2].plot(h.val_recon_mse, label=name)
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].set_title("Validation recon MSE"); axes[2].set_xlabel("epoch"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    labels = [f"{f:g} Hz" for f in FREQUENCIES_HZ]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion — {name}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_reconstruction_examples(
    recon: torch.Tensor,
    clean: torch.Tensor,
    name: str,
    out_path: Path,
    n: int = 4,
) -> None:
    n = min(n, recon.size(0))
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(clean[i].numpy(), label="clean", linewidth=2)
        ax.plot(recon[i].numpy(), label="pred", linestyle="--")
        ax.set_title(f"sample {i}")
        if i == 0:
            ax.legend()
    fig.suptitle(f"Reconstruction — {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def per_sigma_recon(
    recon: torch.Tensor, clean: torch.Tensor, sigma: torch.Tensor
) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for s in NOISE_LEVELS:
        mask = torch.isclose(sigma, torch.tensor(s), atol=1e-4)
        if mask.any():
            mse = nn.functional.mse_loss(recon[mask], clean[mask]).item()
            out[s] = mse
    return out


def summarize(model_name: str, logits, y, recon, clean, sigma) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    mse = nn.functional.mse_loss(recon, clean).item()
    per_sig = per_sigma_recon(recon, clean, sigma)
    print(f"[{model_name}] test acc={acc:.3f} | recon MSE={mse:.4f} | per-σ MSE={per_sig}")
    return {"acc": acc, "mse": mse, **{f"mse_sigma_{int(k*100)}": v for k, v in per_sig.items()}}
