"""Command-line interface: `python -m diabetes_classifier {cv,tune,train} ...`."""

import argparse
import dataclasses
import logging
from pathlib import Path

from diabetes_classifier.config import Config, load_config, save_config

DEFAULT_CONFIG = Path("configs/default.yaml")
OVERRIDES = (
    "seed",
    "max_epochs",
    "learning_rate",
    "hidden_sizes",
    "dropout",
    "ensemble_size",
    "cv_folds",
    "cv_repeats",
    "output_dir",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, help="random seed")
    common.add_argument("--max-epochs", type=int, help="maximum training epochs")
    common.add_argument("--lr", dest="learning_rate", type=float, help="learning rate")
    common.add_argument("--hidden-sizes", type=int, nargs="+", help="hidden layer widths")
    common.add_argument("--dropout", type=float, help="dropout probability")
    common.add_argument("--ensemble-size", type=int, help="networks in the ensemble")
    common.add_argument("--cv-folds", type=int, help="folds per CV repeat")
    common.add_argument("--cv-repeats", type=int, help="CV repeats")
    common.add_argument("--output-dir", help="folder for results")
    common.add_argument("-v", "--verbose", action="store_true", help="log every epoch")

    parser = argparse.ArgumentParser(
        prog="python -m diabetes_classifier",
        description="Diabetes classifier: cross-validate, tune or train and test.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    cv = commands.add_parser(
        "cv", parents=[common], help="compare configs and models with repeated CV"
    )
    cv.add_argument("--config", type=Path, nargs="+", default=[DEFAULT_CONFIG])

    tune = commands.add_parser(
        "tune", parents=[common], help="search network hyperparameters with Optuna"
    )
    tune.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    tune.add_argument("--trials", type=int, default=60, help="number of Optuna trials")
    tune.add_argument("--timeout", type=float, help="stop the search after this many seconds")
    tune.add_argument("--out", type=Path, default=Path("configs/tuned.yaml"), help="tuned config")

    train = commands.add_parser(
        "train", parents=[common], help="train on the development set and evaluate on the test set"
    )
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--pairplot", action="store_true", help="also draw the feature pairplot")
    train.add_argument("--show", action="store_true", help="display figures after the run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_paths = args.config if isinstance(args.config, list) else [args.config]
    configs = [_with_overrides(load_config(path), args) for path in config_paths]

    output_dir = Path(configs[0].output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / f"{args.command}.log", mode="w", encoding="utf-8"),
        ],
    )
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "cv":
        from diabetes_classifier.experiment import plot_cv_results, run_cv

        for config in configs:
            run_cv(config)
        logging.info("Saved %s", plot_cv_results(output_dir))

    elif args.command == "tune":
        from diabetes_classifier.tune import save_study, tune

        tuned, study = tune(configs[0], args.trials, args.timeout)
        save_config(tuned, args.out)
        logging.info("Saved tuned config to %s", args.out)
        logging.info("Saved trial log to %s", save_study(study, output_dir))

    else:
        from diabetes_classifier.experiment import run_final

        run_final(configs[0], pairplot=args.pairplot)
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()


def _with_overrides(config: Config, args: argparse.Namespace) -> Config:
    overrides = {key: value for key in OVERRIDES if (value := getattr(args, key, None)) is not None}
    return dataclasses.replace(config, **overrides)


if __name__ == "__main__":
    main()
