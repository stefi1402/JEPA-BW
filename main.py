def main():
    import argparse
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader


     # 1. DATA GENERATION

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
        print(f"Val:   {args.num_val}")
        print(f"Test:  {args.num_test}")


    
    # 2. TRANSFORMER MODELn

    class CoordinateTransformer(nn.Module):
        def __init__(
            self,
            d = 10,
            input_length = 80,
            prediction_length = 5,
            embed_dim = 64,
            num_heads = 4,
            num_layers = 1,
            dropout = 0.1,
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
            x shape:
                (batch_size, input_length, d*d)

            output shape:
                (batch_size, prediction_length, 2)
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


   
    # 3. DATASET CLASS

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


   
    # 4. TRAINING

    def train(args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Saved best model to {checkpoint_dir / 'best_model.pt'}")


    # 5. TESTING

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


    

    def build_parser():
        parser = argparse.ArgumentParser()

        subparsers = parser.add_subparsers(dest="command", required=True)

        # generate

        generate_parser = subparsers.add_parser("generate")

        generate_parser.add_argument("--output-dir", type=str, default="data")

        generate_parser.add_argument("--d", type=int, default=10)
        generate_parser.add_argument("--T", type=int, default=100)

        generate_parser.add_argument("--p", type=float, default=0.1)
        generate_parser.add_argument("--k", type=int, default=1)

        generate_parser.add_argument("--num-train", type=int, default=10000)
        generate_parser.add_argument("--num-val", type=int, default=2000)
        generate_parser.add_argument("--num-test", type=int, default=2000)

        # -------------------------
        # transformer
        # -------------------------

        transformer_parser = subparsers.add_parser("transformer")

        transformer_parser.add_argument("--d", type=int, default=10)
        transformer_parser.add_argument("--input-length", type=int, default=50)
        transformer_parser.add_argument("--prediction-length", type=int, default=50)

        transformer_parser.add_argument("--embed-dim", type=int, default=64)
        transformer_parser.add_argument("--num-heads", type=int, default=4)
        transformer_parser.add_argument("--num-layers", type=int, default=2)
        transformer_parser.add_argument("--dropout", type=float, default=0.1)

        transformer_parser.add_argument("--batch-size", type=int, default=64)

        # -------------------------
        # train
        # -------------------------

        train_parser = subparsers.add_parser("train")

        train_parser.add_argument("--data-dir", type=str, default="data")
        train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

        train_parser.add_argument("--d", type=int, default=10)
        train_parser.add_argument("--input-length", type=int, default=50)
        train_parser.add_argument("--prediction-length", type=int, default=50)

        train_parser.add_argument("--embed-dim", type=int, default=64)
        train_parser.add_argument("--num-heads", type=int, default=4)
        train_parser.add_argument("--num-layers", type=int, default=2)
        train_parser.add_argument("--dropout", type=float, default=0.1)

        train_parser.add_argument("--batch-size", type=int, default=64)
        train_parser.add_argument("--epochs", type=int, default=50)
        train_parser.add_argument("--lr", type=float, default=1e-3)
        train_parser.add_argument("--weight-decay", type=float, default=1e-4)

        # -------------------------
        # test
        # -------------------------

        test_parser = subparsers.add_parser("test")

        test_parser.add_argument("--data-dir", type=str, default="data")
        test_parser.add_argument(
            "--checkpoint-path",
            type=str,
            default="checkpoints/best_model.pt",
        )

        test_parser.add_argument("--batch-size", type=int, default=64)

        return parser


    def main():
        parser = build_parser()
        args = parser.parse_args()

        if args.command == "generate":
            generate_dataset(args)

        elif args.command == "transformer":
            run_transformer_check(args)

        elif args.command == "train":
            train(args)

        elif args.command == "test":
            test(args)



if __name__ == "__main__":
    main()

