"""Command-line interface: `python -m diabetes_classifier {cv,tune,train,predict} ...`."""

import argparse
import dataclasses
import logging
from pathlib import Path

import yaml

from diabetes_classifier.config import Config, load_config, save_config

DEFAULT_CONFIG = Path("configs/default.yaml")
DEFAULT_MODEL = Path("results/tuned/model.joblib")
FEATURE_FLAGS = {
    "Pregnancies": ("--pregnancies", "number of pregnancies"),
    "Glucose": ("--glucose", "plasma glucose, 2 h oral glucose tolerance test (mg/dl)"),
    "BloodPressure": ("--blood-pressure", "diastolic blood pressure (mm Hg)"),
    "SkinThickness": ("--skin-thickness", "triceps skin-fold thickness (mm)"),
    "Insulin": ("--insulin", "2-hour serum insulin (µU/ml)"),
    "BMI": ("--bmi", "body mass index (kg/m²)"),
    "DiabetesPedigreeFunction": ("--pedigree", "diabetes pedigree function"),
    "Age": ("--age", "age (years)"),
}
REQUIRED_FEATURES = ("Age", "DiabetesPedigreeFunction")
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
    """Parse the command line (a `Namespace` with `command` and that command's options)."""
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

    predict = commands.add_parser(
        "predict",
        help="predict diabetes for new patients with a saved model",
        description="Give either --input (a CSV with the eight feature columns) or the patient's "
        "values as flags. Omitted or 0 values for glucose, blood pressure, skin thickness, "
        "insulin and BMI are treated as not measured.",
    )
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="saved model file")
    predict.add_argument("--input", type=Path, help="CSV of patients to score")
    predict.add_argument("--output", type=Path, help="write predictions to this CSV")
    for column, (flag, description) in FEATURE_FLAGS.items():
        predict.add_argument(flag, dest=column, type=float, help=description)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the command in `argv`; bad input exits with a one-line error (see -v for more)."""
    args = parse_args(argv)
    try:
        _run(args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        if getattr(args, "verbose", False):
            raise
        raise SystemExit(f"error: {error}") from error


def _run(args: argparse.Namespace) -> None:
    if args.command == "predict":
        _predict(args)
        return

    config_paths = args.config if isinstance(args.config, list) else [args.config]
    configs = [_with_overrides(load_config(path), args) for path in config_paths]
    names = [config.name for config in configs]
    if len(set(names)) < len(names):
        raise ValueError(f"Configs must have different names, got {names}")

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
        from diabetes_classifier.experiment import plot_cv_results, run_cv, write_cv_summary

        for config in configs:
            run_cv(config)
        logging.info("Saved %s", write_cv_summary(output_dir))
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


def _predict(args: argparse.Namespace) -> None:
    import pandas as pd

    from diabetes_classifier.predict import Predictor

    if args.input is not None:
        if any(getattr(args, column) is not None for column in FEATURE_FLAGS):
            raise ValueError("Give either --input or the patient's values as flags, not both")
        features = pd.read_csv(args.input)
    else:
        missing = [
            FEATURE_FLAGS[column][0]
            for column in REQUIRED_FEATURES
            if getattr(args, column) is None
        ]
        if missing:
            raise SystemExit(f"Give --input or at least {' and '.join(missing)}")
        features = pd.DataFrame(
            [{column: getattr(args, column) or 0.0 for column in FEATURE_FLAGS}]
        )

    try:
        predictor = Predictor.load(args.model)
    except TypeError as error:
        raise ValueError(str(error)) from error
    result = features.assign(
        probability=predictor.predict_proba(features).round(3),
        prediction=["diabetes" if p else "no diabetes" for p in predictor.predict(features)],
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} predictions to {args.output}")
    elif len(result) == 1:
        patient = result.iloc[0]
        print(patient.map(lambda v: f"{v:g}" if isinstance(v, float) else v).to_string())
    else:
        print(result.to_string(index=False))
    print(
        f"Model: {predictor.model_name}, threshold {predictor.threshold:.3f}. "
        "Research project only, not for clinical use."
    )


def _with_overrides(config: Config, args: argparse.Namespace) -> Config:
    overrides = {key: value for key in OVERRIDES if (value := getattr(args, key, None)) is not None}
    return dataclasses.replace(config, **overrides)


if __name__ == "__main__":
    main()
