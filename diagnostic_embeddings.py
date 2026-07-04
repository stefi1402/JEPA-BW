"""Diagnostic: does the model's prediction depend on the object's recent direction of motion?"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data import BWSequenceDataset
from model import load_model


def check_direction_sensitivity(data_dir="data", checkpoint_path="checkpoints/best_model.pt",
                                 lookback=5):
    model, checkpoint, device = load_model(checkpoint_path)
    model.eval()

    input_length = checkpoint["input_length"]
    prediction_length = checkpoint["prediction_length"]

    dataset = BWSequenceDataset(
        path=Path(data_dir) / "val.npz",
        input_length=input_length,
        prediction_length=prediction_length,
    )

    # We also need raw coords to compute "recent direction" before flattening/reshaping
    raw = np.load(Path(data_dir) / "val.npz")
    coords = raw["coords"]  # (num_sequences, T, 2)

    # Recent direction: displacement between (input_length - 1) and (input_length - 1 - lookback)
    recent_start = coords[:, input_length - 1 - lookback]
    recent_end = coords[:, input_length - 1]
    recent_disp = recent_end - recent_start  # (num_sequences, 2): (d_row, d_col)

    def classify(disp):
        d_row, d_col = disp
        if abs(d_row) < 0.5 and abs(d_col) < 0.5:
            return "stationary"
        if abs(d_row) >= abs(d_col):
            return "down" if d_row > 0 else "up"
        return "right" if d_col > 0 else "left"

    directions = [classify(recent_disp[i]) for i in range(len(recent_disp))]

    # Get model predictions for the whole val set
    x_all = dataset.x.to(device)
    with torch.no_grad():
        pred_all = model(x_all).cpu().numpy()  # (num_sequences, prediction_length, 2)

    # Predicted displacement: first predicted position minus last observed position
    last_observed = coords[:, input_length - 1]  # (num_sequences, 2)
    pred_first_step = pred_all[:, 0, :]  # (num_sequences, 2)
    pred_disp = pred_first_step - last_observed  # (num_sequences, 2)

    # Group by direction category
    grouped = defaultdict(list)
    for i, direction in enumerate(directions):
        grouped[direction].append(pred_disp[i])

    print(f"{'Direction':<12} {'Count':<8} {'Mean pred d_row':<18} {'Mean pred d_col':<18}")
    for direction, disps in sorted(grouped.items()):
        disps = np.array(disps)
        print(f"{direction:<12} {len(disps):<8} {disps[:, 0].mean():<18.4f} {disps[:, 1].mean():<18.4f}")

    print("\nIf the model is sensitive to direction, 'down' should show positive mean d_row,")
    print("'up' negative mean d_row, 'right' positive mean d_col, 'left' negative mean d_col.")
    print("If all groups look similar (near zero, no clear pattern), the model is likely")
    print("ignoring recent motion and just predicting 'stay at last position' regardless of input.")


if __name__ == "__main__":
    check_direction_sensitivity()