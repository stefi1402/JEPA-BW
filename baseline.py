"""Baseline: predict the object stays at its last observed position."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def evaluate_baseline(data_dir, input_length, prediction_length):
    data = np.load(Path(data_dir) / "val.npz")
    coords = data["coords"]  # shape: (num_sequences, T, 2)

    last_pos = coords[:, input_length - 1]  # (num_sequences, 2)
    y = coords[:, input_length:input_length + prediction_length]  # (num_sequences, prediction_length, 2)

    # Repeat the last observed position for every future timestep
    pred = np.repeat(last_pos[:, None, :], prediction_length, axis=1)

    pred = torch.tensor(pred, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(pred, y).item()

    distance = torch.sqrt(((pred - y) ** 2).sum(dim=-1))
    mean_coordinate_error = distance.mean().item()

    print(f"Baseline (stay at last position)")
    print(f"Loss: {loss:.4f}")
    print(f"Mean coordinate error: {mean_coordinate_error:.4f} pixels")


if __name__ == "__main__":
    evaluate_baseline(data_dir="data", input_length=50, prediction_length=50)