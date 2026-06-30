"""Command-line entry point for the JEPA-BW project."""

from cli import build_parser
from data import BWSequenceDataset, generate_dataset, generate_one_sequence
from evaluate import test
from model import CoordinateTransformer, run_transformer_check
from train import train

__all__ = [
    "BWSequenceDataset",
    "CoordinateTransformer",
    "build_parser",
    "generate_dataset",
    "generate_one_sequence",
    "main",
    "run_transformer_check",
    "test",
    "train",
]


def main() -> None:
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
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()