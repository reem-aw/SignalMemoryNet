"""CLI entry point: train MLP / RNN / LSTM on the synthetic sine task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config.config import TrainConfig
from .data import make_dataloaders
from .evaluate import (
    collect_predictions,
    confusion_matrix,
    plot_confusion,
    plot_history,
    plot_reconstruction_examples,
    summarize,
)
from .models.lstm import LSTMModel
from .models.mlp import MLPModel
from .models.rnn import RNNModel
from .train import History, train_one

MODEL_BUILDERS = {
    "mlp": lambda cfg: MLPModel(hidden=cfg.mlp_hidden),
    "rnn": lambda cfg: RNNModel(hidden_size=cfg.hidden_size),
    "lstm": lambda cfg: LSTMModel(hidden_size=cfg.hidden_size),
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(model_names, cfg: TrainConfig, out_dir: Path):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_dataloaders(
        cfg.train_records_per_freq,
        cfg.val_records_per_freq,
        cfg.test_records_per_freq,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        window_stride=cfg.window_stride,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    histories: dict[str, History] = {}
    summaries: dict[str, dict] = {}

    for name in model_names:
        print(f"\n=== Training {name.upper()} ===")
        model = MODEL_BUILDERS[name](cfg)
        history = train_one(model, train_loader, val_loader, cfg, device=device)
        histories[name] = history
        torch.save(model.state_dict(), ckpt_dir / f"{name}.pt")

        logits, y, recon, clean, sigma = collect_predictions(model, test_loader, device)
        summaries[name] = summarize(name, logits, y, recon, clean, sigma)
        cm = confusion_matrix(y, logits.argmax(dim=1), n_classes=logits.size(1))
        plot_confusion(cm, name, fig_dir / f"confusion_{name}.png")
        plot_reconstruction_examples(recon, clean, name, fig_dir / f"recon_{name}.png")

    plot_history(histories, fig_dir / "training_curves.png")
    (out_dir / "metrics.json").write_text(json.dumps(summaries, indent=2))
    print("\nSaved figures to", fig_dir)
    print("Metrics:", json.dumps(summaries, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_BUILDERS) + ["all"], default="all")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("report"))
    args = ap.parse_args()
    cfg = TrainConfig(epochs=args.epochs)
    names = list(MODEL_BUILDERS) if args.model == "all" else [args.model]
    run(names, cfg, args.out)


if __name__ == "__main__":
    main()
