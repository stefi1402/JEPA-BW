"""Argument parser for the JEPA-BW command-line interface."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output-dir", type=str, default="data")
    generate_parser.add_argument("--d", type=int, default=10)
    generate_parser.add_argument("--T", type=int, default=100)
    generate_parser.add_argument("--p", type=float, default=0.2)
    generate_parser.add_argument("--k", type=int, default=1)
    generate_parser.add_argument("--num-train", type=int, default=10000)
    generate_parser.add_argument("--num-val", type=int, default=2000)
    generate_parser.add_argument("--num-test", type=int, default=2000)
    generate_parser.add_argument("--seed", type=int, default=None)

    transformer_parser = subparsers.add_parser("transformer")
    transformer_parser.add_argument("--d", type=int, default=10)
    transformer_parser.add_argument("--input-length", type=int, default=50)
    transformer_parser.add_argument("--prediction-length", type=int, default=50)
    transformer_parser.add_argument("--embed-dim", type=int, default=64)
    transformer_parser.add_argument("--num-heads", type=int, default=4)
    transformer_parser.add_argument("--num-layers", type=int, default=2)
    transformer_parser.add_argument("--dropout", type=float, default=0.1)
    transformer_parser.add_argument("--batch-size", type=int, default=64)

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
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=15)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes. >0 only helps if loading is a bottleneck.",
    )

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--data-dir", type=str, default="data")
    test_parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/best_model.pt",
    )
    test_parser.add_argument("--batch-size", type=int, default=64)

    return parser
