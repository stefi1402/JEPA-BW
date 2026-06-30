"""Dataset generation and loading utilities."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_one_sequence(d, T, p, k):
    frames = np.zeros((T, d, d), dtype=np.float32)
    coords = np.zeros((T, 2), dtype=np.float32)

    row = np.random.randint(0, d)
    col = np.random.randint(0, d)

    for t in range(T):
        frames[t, row, col] = 1.0
        coords[t] = [row, col]

        if np.random.rand() < p:
            direction = np.random.choice(["up", "down", "left", "right"])

            if direction == "up":
                row = max(0, row - k)
            elif direction == "down":
                row = min(d - 1, row + k)
            elif direction == "left":
                col = max(0, col - k)
            elif direction == "right":
                col = min(d - 1, col + k)

    return frames, coords


def generate_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_sequences = args.num_train + args.num_val + args.num_test

    all_frames = []
    all_coords = []

    for _ in range(total_sequences):
        frames, coords = generate_one_sequence(
            d=args.d,
            T=args.T,
            p=args.p,
            k=args.k,
        )
        all_frames.append(frames)
        all_coords.append(coords)

    all_frames = np.stack(all_frames)
    all_coords = np.stack(all_coords)

    train_end = args.num_train
    val_end = args.num_train + args.num_val

    np.savez(
        output_dir / "train.npz",
        frames=all_frames[:train_end],
        coords=all_coords[:train_end],
    )
    np.savez(
        output_dir / "val.npz",
        frames=all_frames[train_end:val_end],
        coords=all_coords[train_end:val_end],
    )
    np.savez(
        output_dir / "test.npz",
        frames=all_frames[val_end:],
        coords=all_coords[val_end:],
    )

    print(f"Saved dataset to {output_dir}")
    print(f"Train: {args.num_train}")
    print(f"Val: {args.num_val}")
    print(f"Test: {args.num_test}")


class BWSequenceDataset(Dataset):
    def __init__(self, path, input_length, prediction_length):
        data = np.load(path)
        frames = data["frames"]
        coords = data["coords"]

        self.x = frames[:, :input_length]
        self.y = coords[:, input_length:input_length + prediction_length]

        num_sequences, input_length, d, _ = self.x.shape
        self.x = self.x.reshape(num_sequences, input_length, d * d)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
