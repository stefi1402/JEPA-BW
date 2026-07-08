"""Plot predicted vs. true trajectories on the grid, for sanity-checking a model.

Usage (after training):
    python3 visualize.py --data-dir data --checkpoint-path checkpoints/best_model.pt

Also importable from train.py to save a snapshot every few epochs, so you
can flip through epoch_010.png -> epoch_050.png -> epoch_100.png and see
whether the red (predicted) path visibly starts tracking the green (true)
path over the course of training, rather than staring at loss numbers.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; just write image files
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import load_model


def render_predictions(
    model,
    device,
    data_dir,
    input_length,
    prediction_length,
    d,
    num_examples=6,
    seed=0,
    title=None,
):
    """Build a figure of `num_examples` sample sequences: observed path (blue),
    true future (green), predicted future (red). Returns the matplotlib Figure;
    caller decides whether to save/show/close it.
    """
    raw = np.load(Path(data_dir) / "val.npz")
    frames = raw["frames"]  # (N, T, d, d)
    coords = raw["coords"]  # (N, T, 2)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), size=min(num_examples, len(coords)), replace=False)

    x = frames[idx, :input_length].reshape(len(idx), input_length, d * d)
    x = torch.tensor(x, dtype=torch.float32).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy()  # (num_examples, prediction_length, 2)
    model.train(was_training)

    observed = coords[idx, :input_length]
    true_future = coords[idx, input_length:input_length + prediction_length]

    cols = min(3, len(idx))
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes = axes.reshape(-1)

    for i, ax in enumerate(axes):
        if i >= len(idx):
            ax.axis("off")
            continue

        obs = observed[i]
        true_f = true_future[i]
        pred_f = pred[i]

        ax.set_xlim(-0.5, d - 0.5)
        ax.set_ylim(d - 0.5, -0.5)  # row 0 at top, like the array layout
        ax.set_xticks(range(d))
        ax.set_yticks(range(d))
        ax.grid(True, linewidth=0.3, color="lightgray")
        ax.set_aspect("equal")

        # note: coords are (row, col); plot as (x=col, y=row)
        ax.plot(obs[:, 1], obs[:, 0], "-", color="steelblue", alpha=0.5, linewidth=1.5)
        ax.plot(obs[-1, 1], obs[-1, 0], "s", color="steelblue", markersize=9, label="last observed")
        ax.plot(true_f[:, 1], true_f[:, 0], "-o", color="seagreen", markersize=3, linewidth=1.5, label="true future")
        ax.plot(pred_f[:, 1], pred_f[:, 0], "-o", color="crimson", markersize=3, linewidth=1.5, label="predicted")

        ax.set_title(f"sequence {idx[i]}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-path", default="checkpoints/best_model.pt")
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--output", default="predictions.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, checkpoint, device = load_model(args.checkpoint_path)

    fig = render_predictions(
        model=model,
        device=device,
        data_dir=args.data_dir,
        input_length=checkpoint["input_length"],
        prediction_length=checkpoint["prediction_length"],
        d=checkpoint["d"],
        num_examples=args.num_examples,
        seed=args.seed,
    )
    fig.savefig(args.output, dpi=130)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
