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


def generate_batch(n, d, T, p, k, rng=None):
    """Vectorized equivalent of calling generate_one_sequence n times.

    generate_one_sequence is kept around for single-sequence use (e.g. a
    quick sanity check), but generating a full dataset by calling it in a
    Python for-loop means T * total_sequences pure-Python iterations
    (e.g. 100 * 14000 = 1.4M) just to build a training set. Here the
    T-length loop stays in Python, but every iteration operates on all n
    sequences at once via numpy, so the cost is ~T vectorized steps
    instead of ~T * n scalar ones.

    Note: this consumes randomness in a different order than repeated
    generate_one_sequence calls, so it won't reproduce bit-identical
    datasets for the same seed, but the generating distribution is
    identical (uniform start position, independent per-step move
    probability p, uniform choice of the 4 directions).
    """
    if rng is None:
        rng = np.random.default_rng()

    frames = np.zeros((n, T, d, d), dtype=np.float32)
    coords = np.zeros((n, T, 2), dtype=np.float32)

    row = rng.integers(0, d, size=n)
    col = rng.integers(0, d, size=n)
    idx = np.arange(n)

    for t in range(T):
        frames[idx, t, row, col] = 1.0
        coords[idx, t, 0] = row
        coords[idx, t, 1] = col

        moves = rng.random(n) < p
        direction = rng.integers(0, 4, size=n)  # 0=up 1=down 2=left 3=right

        up = moves & (direction == 0)
        down = moves & (direction == 1)
        left = moves & (direction == 2)
        right = moves & (direction == 3)

        row[up] = np.maximum(0, row[up] - k)
        row[down] = np.minimum(d - 1, row[down] + k)
        col[left] = np.maximum(0, col[left] - k)
        col[right] = np.minimum(d - 1, col[right] + k)

    return frames, coords


def generate_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_sequences = args.num_train + args.num_val + args.num_test
    rng = np.random.default_rng(getattr(args, "seed", None))

    all_frames, all_coords = generate_batch(
        n=total_sequences,
        d=args.d,
        T=args.T,
        p=args.p,
        k=args.k,
        rng=rng,
    )

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
