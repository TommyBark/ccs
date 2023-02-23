"""Main entry point for `elk`."""

from .extraction import ExtractionConfig
from .list import list_runs
from .training import RunConfig
from pathlib import Path
from simple_parsing import ArgumentParser


def run():
    parser = ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract hidden states from a model.",
    )
    extract_parser.add_arguments(ExtractionConfig, dest="extraction")
    extract_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Path to save hidden states to.",
        required=True,
    )

    elicit_parser = subparsers.add_parser(
        "elicit",
        help=(
            "Extract and train a set of ELK reporters "
            "on hidden states from `elk extract`. "
        ),
        conflict_handler="resolve",
    )
    elicit_parser.add_arguments(RunConfig, dest="run")
    elicit_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Path to save checkpoints to.",
        required=True,
    )

    subparsers.add_parser(
        "eval", help="Evaluate a set of ELK reporters generated by `elk train`."
    )
    subparsers.add_parser("list", help="List all cached runs.")
    args = parser.parse_args()

    # `elk list` is a special case
    if args.command == "list":
        list_runs(args)
        return

    # Import here and not at the top to speed up `elk list`
    from .extraction.extraction import extract_to_disk
    from .training.train import train

    if args.command == "extract":
        extract_to_disk(args.extraction, args.output)
    elif args.command == "elicit":
        train(args.run, args.output)

    elif args.command == "eval":
        # TODO: Implement evaluation script
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    run()
