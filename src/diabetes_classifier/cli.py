"""Command-line entry point: `python -m diabetes_classifier` or `diabetes-classifier`."""

import argparse
import dataclasses
import logging
from pathlib import Path

from diabetes_classifier.config import load_config
from diabetes_classifier.experiment import run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the diabetes classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML config")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--epochs", type=int, help="number of training epochs")
    parser.add_argument("--lr", dest="learning_rate", type=float, help="SGD learning rate")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", help="hidden layer widths")
    parser.add_argument("--dropout", type=float, help="dropout probability")
    parser.add_argument("--output-dir", help="folder for metrics, figures and trial log")
    parser.add_argument("--pairplot", action="store_true", help="also draw the feature pairplot")
    parser.add_argument("--show", action="store_true", help="display figures after the run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    overrides = {
        key: value
        for key in ("seed", "epochs", "learning_rate", "hidden_sizes", "dropout", "output_dir")
        if (value := getattr(args, key)) is not None
    }
    config = dataclasses.replace(config, **overrides)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8"),
        ],
    )

    run_experiment(config, pairplot=args.pairplot)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
