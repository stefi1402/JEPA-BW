"""Training loop for the coordinate transformer."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import BWSequenceDataset
from model import CoordinateTransformer


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print("Args:", vars(args))

    train_dataset = BWSequenceDataset(
        path=Path(args.data_dir) / "train.npz",
        input_length=args.input_length,
        prediction_length=args.prediction_length,
    )
    val_dataset = BWSequenceDataset(
        path=Path(args.data_dir) / "val.npz",
        input_length=args.input_length,
        prediction_length=args.prediction_length,
    )

    print(f"Train sequences: {len(train_dataset)} | Val sequences: {len(val_dataset)}")
    print(f"Batches per epoch: {len(train_dataset) // args.batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = CoordinateTransformer(
        d=args.d,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "d": args.d,
                "input_length": args.input_length,
                "prediction_length": args.prediction_length,
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {args.patience} epochs)"
                )
                break

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved best model to {checkpoint_dir / 'best_model.pt'}")