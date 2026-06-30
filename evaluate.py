"""Evaluation code for a saved coordinate transformer checkpoint."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import BWSequenceDataset
from model import CoordinateTransformer


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    model = CoordinateTransformer(
        d=checkpoint["d"],
        input_length=checkpoint["input_length"],
        prediction_length=checkpoint["prediction_length"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = BWSequenceDataset(
        path=Path(args.data_dir) / "test.npz",
        input_length=checkpoint["input_length"],
        prediction_length=checkpoint["prediction_length"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    loss_fn = nn.SmoothL1Loss()

    total_loss = 0.0
    total_distance = 0.0
    total_points = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

            distance = torch.sqrt(((pred - y) ** 2).sum(dim=-1))
            total_distance += distance.sum().item()
            total_points += distance.numel()

    test_loss = total_loss / len(test_loader)
    mean_coordinate_error = total_distance / total_points

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean coordinate error: {mean_coordinate_error:.4f} pixels")
