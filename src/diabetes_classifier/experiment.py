"""End-to-end experiment: data, training, evaluation, figures and trial log."""

import csv
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from diabetes_classifier import plots
from diabetes_classifier.config import Config
from diabetes_classifier.data import load_dataset, split_and_scale
from diabetes_classifier.evaluate import evaluate, predict_proba
from diabetes_classifier.model import DiabetesClassifier
from diabetes_classifier.train import train

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every random number generator the run uses, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(config: Config, pairplot: bool = False) -> dict:
    """Run one experiment and write its outputs to `config.output_dir`.

    Outputs: `metrics.json`, figures in `figures/`, and one appended row in `trials.csv`.
    Returns the trial row.
    """
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(config.data_path)
    splits = split_and_scale(df, config.test_size, config.val_size, config.seed)
    logger.info(
        "Samples: %d train / %d validation / %d test",
        len(splits.y_train),
        len(splits.y_val),
        len(splits.y_test),
    )

    model = DiabetesClassifier(
        splits.n_features, config.hidden_sizes, config.dropout, config.dropout_layers
    )
    history = train(model, splits, config.epochs, config.learning_rate, config.log_every)

    train_metrics = evaluate(model, splits.X_train, splits.y_train)
    test_metrics = evaluate(model, splits.X_test, splits.y_test)
    for name, value in test_metrics.as_dict().items():
        if name != "confusion_matrix":
            logger.info("test %-9s %.2f%%", name, 100 * value)

    figures = {
        "loss_curves": plots.loss_curves(history),
        "roc_curve": plots.roc(
            splits.y_test.numpy(), predict_proba(model, splits.X_test), test_metrics.roc_auc
        ),
        "confusion_matrix": plots.confusion(test_metrics.confusion_matrix),
    }
    if pairplot:
        figures["pairplot"] = plots.pairplot(df)
    for name, fig in figures.items():
        fig.savefig(figures_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    logger.info("Saved figures to %s", figures_dir)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(config),
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
                "train": train_metrics.as_dict(),
                "test": test_metrics.as_dict(),
            },
            f,
            indent=2,
        )

    trial = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": config.seed,
        "hidden_sizes": "-".join(map(str, config.hidden_sizes)),
        "dropout": config.dropout,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "best_epoch": history.best_epoch,
        "best_val_loss": round(history.best_val_loss, 4),
        "train_accuracy": round(train_metrics.accuracy, 4),
        "test_accuracy": round(test_metrics.accuracy, 4),
        "precision": round(test_metrics.precision, 4),
        "recall": round(test_metrics.recall, 4),
        "f1": round(test_metrics.f1, 4),
        "roc_auc": round(test_metrics.roc_auc, 4),
    }
    _append_trial(output_dir / "trials.csv", trial)
    return trial


def _append_trial(path: Path, trial: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial))
        if is_new:
            writer.writeheader()
        writer.writerow(trial)
