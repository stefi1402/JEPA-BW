"""Sanity-check the generated dataset by rendering actual raw frames.

Unlike visualize.py (which plots trajectories as coordinate scatter/lines),
this shows the literal black & white grids the model receives as input,
so you can eyeball things like: is there exactly one lit pixel per frame,
does it move by the expected step size, are frames actually binary, etc.

Usage:
    python3 inspect_frames.py --data-dir data --split train --sequence 0 --num-frames 12
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def inspect(data_dir, split, sequence_idx, num_frames, start_frame, stride, output_path):
    raw = np.load(Path(data_dir) / f"{split}.npz")
    frames = raw["frames"]  # (N, T, d, d)
    coords = raw["coords"]  # (N, T, 2)

    seq_frames = frames[sequence_idx]  # (T, d, d)
    seq_coords = coords[sequence_idx]  # (T, 2)
    T = seq_frames.shape[0]

    # --- Consistency checks, printed so you don't have to eyeball them ---
    pixel_sums = seq_frames.reshape(T, -1).sum(axis=1)
    is_binary = np.all((seq_frames == 0) | (seq_frames == 1))
    print(f"Sequence {sequence_idx} ({T} total frames):")
    print(f"  Frames with exactly one lit pixel: {(pixel_sums == 1).sum()} / {T}")
    if not np.all(pixel_sums == 1):
        bad = np.where(pixel_sums != 1)[0]
        print(f"  WARNING: frames with != 1 lit pixel at indices: {bad[:10]}{'...' if len(bad) > 10 else ''}")
    print(f"  All values strictly 0 or 1: {is_binary}")

    # Does the lit pixel's location actually match the stored coords?
    lit_positions = np.array([np.argwhere(seq_frames[t] == 1)[0] if pixel_sums[t] == 1 else [-1, -1]
                               for t in range(T)])
    mismatches = np.sum(np.any(lit_positions != seq_coords.astype(int), axis=1) & (pixel_sums == 1))
    print(f"  Frames where lit pixel matches stored coords: {T - mismatches} / {T}")

    step_sizes = np.sqrt(((np.diff(seq_coords, axis=0)) ** 2).sum(axis=-1))
    print(f"  Step sizes between consecutive frames: min={step_sizes.min():.1f}, "
          f"max={step_sizes.max():.1f}, moves > 0: {(step_sizes > 0).sum()}/{T - 1}")

    # --- Render frames starting at start_frame, stepping by `stride` ---
    # Default stride=1 shows genuinely consecutive frames (t, t+1, t+2, ...)
    # so you can see whether/where it moves frame-to-frame, rather than a
    # handful of samples spread thinly across the whole sequence.
    show_idx = start_frame + np.arange(num_frames) * stride
    show_idx = show_idx[show_idx < T]
    if len(show_idx) < num_frames:
        print(f"  (only {len(show_idx)} frames fit before the sequence ends at t={T - 1})")

    cols = min(6, len(show_idx))
    rows = int(np.ceil(len(show_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, t in enumerate(show_idx):
        ax = axes[ax_i]
        ax.imshow(seq_frames[t], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"t={t}\n({int(seq_coords[t, 0])},{int(seq_coords[t, 1])})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax_i in range(len(show_idx), len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle(f"{split}.npz — sequence {sequence_idx}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--start-frame", type=int, default=0,
                         help="First frame index to show (default: 0).")
    parser.add_argument("--stride", type=int, default=1,
                         help="Step between shown frames. 1 = consecutive frames "
                              "(t, t+1, t+2, ...); higher values space them out.")
    parser.add_argument("--output", default="frames_inspection.png")
    args = parser.parse_args()
    inspect(args.data_dir, args.split, args.sequence, args.num_frames,
             args.start_frame, args.stride, args.output)


if __name__ == "__main__":
    main()
