"""Transformer model used to predict future coordinates."""

import torch
import torch.nn as nn


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
            torch.randn(1, input_length, embed_dim)
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

    def forward(self, x):
        """
        x shape: (batch_size, input_length, d*d)
        output shape: (batch_size, prediction_length, 2)
        """
        x = self.frame_embedding(x)
        x = x + self.time_embedding
        z = self.transformer(x)

        # Use the final context representation
        z_last = z[:, -1, :]
        coords = self.predictor(z_last)
        coords = coords.view(-1, self.prediction_length, 2)
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
