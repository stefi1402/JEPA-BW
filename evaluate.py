"""Evaluation code for a saved coordinate transformer checkpoint.

Beyond the single averaged error, this also checks two things that a flat
average can hide entirely:

1. Per-horizon error: is the model actually better at predicting 1 step
   ahead than 50 steps ahead (as it should be), and how does it compare
   to the "stay at last position" baseline at *each* horizon, not just on
   average?
2. Trajectory coherence: does the predicted 50-step path move smoothly,
   or does it jump around between consecutive predicted steps in a way
   that couldn't correspond to any real trajectory the object could take?
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import BWSequenceDataset
from model import load_model


def test(args):
    model, checkpoint, device = load_model(args.checkpoint_path)
    model.eval()

    input_length = checkpoint["input_length"]
    prediction_length = checkpoint["prediction_length"]

    test_dataset = BWSequenceDataset(
        path=Path(args.data_dir) / "test.npz",
        input_length=input_length,
        prediction_length=prediction_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    loss_fn = nn.MSELoss()

    total_loss = 0.0
    all_pred = []
    all_y = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            all_pred.append(pred.cpu())
            all_y.append(y.cpu())

    all_pred = torch.cat(all_pred)  # (N, prediction_length, 2)
    all_y = torch.cat(all_y)

    test_loss = total_loss / len(test_loader)
    distance = torch.sqrt(((all_pred - all_y) ** 2).sum(dim=-1))  # (N, prediction_length)
    mean_coordinate_error = distance.mean().item()

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean coordinate error: {mean_coordinate_error:.4f} pixels")

    # --- Baseline for comparison: predict "stay at last observed position" ---
    raw = np.load(Path(args.data_dir) / "test.npz")
    coords = raw["coords"]  # (N, T, 2)
    last_observed = coords[:, input_length - 1]  # (N, 2)
    baseline_pred = np.repeat(last_observed[:, None, :], prediction_length, axis=1)
    baseline_pred = torch.tensor(baseline_pred, dtype=torch.float32)
    y_np = coords[:, input_length:input_length + prediction_length]
    y_np = torch.tensor(y_np, dtype=torch.float32)
    baseline_distance = torch.sqrt(((baseline_pred - y_np) ** 2).sum(dim=-1))  # (N, prediction_length)

    # --- 1. Per-horizon error: model vs. baseline at each future step ---
    model_by_horizon = distance.mean(dim=0).numpy()      # (prediction_length,)
    baseline_by_horizon = baseline_distance.mean(dim=0).numpy()

    print("\nPer-horizon mean coordinate error (model vs. baseline):")
    steps_to_show = sorted(set([0, prediction_length // 4, prediction_length // 2,
                                 3 * prediction_length // 4, prediction_length - 1]))
    for s in steps_to_show:
        print(f"  step {s + 1:>3}: model {model_by_horizon[s]:.4f}  |  baseline {baseline_by_horizon[s]:.4f}")

    if (model_by_horizon > baseline_by_horizon).mean() > 0.5:
        print("  -> Model is WORSE than the 'stay put' baseline on most horizons.")
    else:
        print("  -> Model matches or beats the 'stay put' baseline on most horizons.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, prediction_length + 1), model_by_horizon, "-o", markersize=3, label="model")
    ax.plot(range(1, prediction_length + 1), baseline_by_horizon, "--", label="baseline (stay put)")
    ax.set_xlabel("Prediction horizon (steps ahead)")
    ax.set_ylabel("Mean coordinate error (pixels)")
    ax.set_title("Error vs. prediction horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig("error_by_horizon.png", dpi=130)
    plt.close(fig)
    print("Saved error_by_horizon.png")

    # --- 2. Coherence: how far does the predicted path jump step-to-step? ---
    pred_steps = all_pred[:, 1:, :] - all_pred[:, :-1, :]      # (N, prediction_length-1, 2)
    pred_step_dist = torch.sqrt((pred_steps ** 2).sum(dim=-1))  # per-step jump size

    true_steps = all_y[:, 1:, :] - all_y[:, :-1, :]
    true_step_dist = torch.sqrt((true_steps ** 2).sum(dim=-1))

    d = checkpoint["d"]
    out_of_bounds = ((all_pred < 0) | (all_pred > d - 1)).any(dim=-1).float().mean().item()

    print("\nTrajectory coherence:")
    print(f"  Mean step-to-step jump   | predicted: {pred_step_dist.mean():.4f}  actual: {true_step_dist.mean():.4f}")
    print(f"  Max step-to-step jump    | predicted: {pred_step_dist.max():.4f}  actual: {true_step_dist.max():.4f}")
    print(f"  Fraction of predicted points outside the {d}x{d} grid: {out_of_bounds:.2%}")
    if pred_step_dist.mean() > true_step_dist.max():
        print("  -> Predicted path jumps around far more than any real trajectory could;"
              " likely near-independent per-step outputs rather than a coherent path.")
    else:
        print("  -> Predicted path's step sizes are within the range real trajectories show.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-path", default="checkpoints/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
