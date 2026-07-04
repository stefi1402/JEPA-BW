"""Training loop for the coordinate transformer."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import BWSequenceDataset
from model import CoordinateTransformer


def _model_config(args):
    """Single source of truth for the fields that reconstruct a model."""
    return {
        "d": args.d,
        "input_length": args.input_length,
        "prediction_length": args.prediction_length,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }


def train(args):
    if getattr(args, "seed", None) is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    use_cuda = device.type == "cuda"
    if use_cuda:
        # Let cuDNN pick the fastest conv/attention kernels for our fixed
        # input shapes instead of using generic defaults.
        torch.backends.cudnn.benchmark = True

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

    num_workers = getattr(args, "num_workers", 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    model = CoordinateTransformer(**_model_config(args)).to(device)

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Halve the LR when val loss plateaus, so a fixed --lr doesn't have to
    # be exactly right for the whole run.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 3)
    )

    # Mixed precision only pays off on CUDA; on CPU/MPS it's a no-op path.
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            # Unscale before clipping so the clip threshold is in real
            # gradient units, not scaled ones.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=use_cuda)
                y = y.to(device, non_blocking=use_cuda)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    pred = model(x)
                    loss = loss_fn(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {"model_state_dict": model.state_dict(), **_model_config(args)}
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
