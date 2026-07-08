"""Transformer model used to predict future coordinates."""

from pathlib import Path

import torch
import torch.nn as nn


def load_model(checkpoint_path, device=None):
    """Build a CoordinateTransformer from a saved checkpoint.

    Centralizes the "read config out of checkpoint, build model, load
    weights" dance that used to be duplicated in evaluate.py,
    diagnostic_embeddings.py, and anywhere else a checkpoint is loaded.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
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
    return model, checkpoint, device


class CoordinateTransformer(nn.Module):
    def __init__(
        self,
        d=10,
        input_length=80,
        prediction_length=5,
        embed_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.d = d
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.frame_dim = d * d 

        self.frame_embedding = nn.Linear(self.frame_dim, embed_dim)
        self.time_embedding = nn.Parameter(
            torch.randn(1, input_length, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, prediction_length * 2),
        )

    def _last_position(self, x):
        """Recover the (row, col) of the dot in the last input frame.

        Each frame in x is one-hot over a d*d grid, so the last observed
        position is just the argmax of the last frame, decoded back into
        2D coordinates. No extra data needs to flow through the Dataset /
        train loop for this -- it's already fully determined by x.
        """
        last_frame = x[:, -1, :]  # (batch, d*d)
        flat_idx = last_frame.argmax(dim=-1)  # (batch,)
        row = (flat_idx // self.d).float()
        col = (flat_idx % self.d).float()
        return torch.stack([row, col], dim=-1)  # (batch, 2)

    def forward(self, x):
        """
        x shape: (batch_size, input_length, d*d)
        output shape: (batch_size, prediction_length, 2)
        """
        last_pos = self._last_position(x)  # (batch, 2)

        x = self.frame_embedding(x)
        x = x + self.time_embedding
        z = self.transformer(x)

        # Pool on the last timestep, not the mean over all timesteps.
        #
        # The encoder is bidirectional (no causal mask), so z[:, -1, :]
        # already attends to the whole sequence and is a valid "summary."
        # But mean-pooling over 50+ frames dilutes exactly the signal this
        # task depends on: where the object is *right now* and which way
        # it's currently moving. That recent-motion signal lives in a
        # handful of frames near the end; averaging it in with distant,
        # already-stale positions pushes the model toward "predict the
        # average / stay put" rather than "extrapolate the current
        # trajectory" (see diagnostic_embeddings.py, which exists to check
        # for exactly this failure mode).
        z_last = z[:, -1, :]

        # Predict a *delta* from the last observed position rather than an
        # absolute coordinate. At init (small weights) this makes the
        # model's default prediction equal to the stay-put baseline, and
        # weight decay pulls it back toward that same default instead of
        # fighting against having to relearn "copy the input" from scratch.
        delta = self.predictor(z_last)
        delta = delta.view(-1, self.prediction_length, 2)
        coords = last_pos.unsqueeze(1) + delta
        return coords


def run_transformer_check(args):
    model = CoordinateTransformer(
        d=args.d,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    dummy_x = torch.randn(
        args.batch_size,
        args.input_length,
        args.d * args.d,
    )
    dummy_y = model(dummy_x)

    print(model)
    print()
    print("Input shape: ", dummy_x.shape)
    print("Output shape:", dummy_y.shape)
