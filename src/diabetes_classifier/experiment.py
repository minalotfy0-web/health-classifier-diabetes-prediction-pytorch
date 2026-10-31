"""Experiments: repeated cross-validation and the final test-set evaluation."""

import csv
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold

from diabetes_classifier import plots
from diabetes_classifier.config import Config
from diabetes_classifier.data import (
    TARGET_COLUMN,
    load_dataset,
    prepare_fold,
    split_development_test,
    stratified_split,
)
from diabetes_classifier.evaluate import METRIC_NAMES, choose_threshold, compute_metrics
from diabetes_classifier.models import fit_predict

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every random number generator the run uses, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_cv(config: Config) -> pd.DataFrame:
    """Score every model with repeated stratified k-fold CV on the development set.

    In each fold the training part is split again into train and validation (early
    stopping, threshold); the held-out fold is only scored. The test set is never used.
    Writes `cv/<config>_folds.csv`, updates `cv_summary.csv` and returns the summary.
    """
    set_seed(config.seed)
    development, _ = split_development_test(load_dataset(config.data_path), config)
    splitter = RepeatedStratifiedKFold(
        n_splits=config.cv_folds, n_repeats=config.cv_repeats, random_state=config.seed
    )
    n_folds = config.cv_folds * config.cv_repeats

    rows = []
    for fold, (fit_index, eval_index) in enumerate(
        splitter.split(development, development[TARGET_COLUMN])
    ):
        train_df, val_df = stratified_split(
            development.iloc[fit_index], config.val_fraction_of_development, config.seed + fold
        )
        data = prepare_fold(train_df, val_df, development.iloc[eval_index], config)
        outputs, _ = fit_predict(config, data, seed=config.seed + 1000 * fold)
        for model_name, output in outputs.items():
            threshold = choose_threshold(data.y_val, output.val_probs, config.threshold)
            metrics = compute_metrics(data.y_eval, output.eval_probs, threshold)
            rows.append(
                {
                    "config": config.name,
                    "model": model_name,
                    "fold": fold,
                    "threshold": metrics.threshold,
                    **metrics.scores(),
                }
            )
        logger.info("[%s] fold %d/%d done", config.name, fold + 1, n_folds)

    folds = pd.DataFrame(rows)
    cv_dir = Path(config.output_dir) / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(cv_dir / f"{config.name}_folds.csv", index=False)

    summary = summarise_folds(folds)
    _update_summary(Path(config.output_dir) / "cv_summary.csv", summary, config.name)
    for _, row in summary.iterrows():
        logger.info(
            "[%s] %-22s ROC AUC %.3f ± %.3f | accuracy %.1f%% ± %.1f",
            config.name,
            row["model"],
            row["roc_auc_mean"],
            row["roc_auc_std"],
            100 * row["accuracy_mean"],
            100 * row["accuracy_std"],
        )
    return summary


def summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric per config and model."""
    summary = folds.groupby(["config", "model"], sort=False)[list(METRIC_NAMES)].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("roc_auc_mean", ascending=False)


def plot_cv_results(output_dir: str | Path) -> Path:
    """Draw the comparison of every CV run saved in `output_dir/cv`."""
    cv_dir = Path(output_dir) / "cv"
    folds = pd.concat(pd.read_csv(path) for path in sorted(cv_dir.glob("*_folds.csv")))
    path = Path(output_dir) / "figures" / "cv_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plots.cv_comparison(folds)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_final(config: Config, pairplot: bool = False) -> dict[str, dict]:
    """Train on the development set and evaluate every model once on the test set.

    Outputs go to `<output_dir>/<config name>/`: metrics.json, test_results.csv and
    figures. The primary model (first in `config.models`) is appended to trials.csv.
    """
    set_seed(config.seed)
    df = load_dataset(config.data_path)
    development, test = split_development_test(df, config)
    train_df, val_df = stratified_split(
        development, config.val_fraction_of_development, config.seed
    )
    data = prepare_fold(train_df, val_df, test, config)
    logger.info(
        "Samples: %d train / %d validation / %d test | %d features",
        len(data.y_train),
        len(data.y_val),
        len(data.y_eval),
        data.n_features,
    )

    outputs, histories = fit_predict(config, data, seed=config.seed)
    results = {}
    for model_name, output in outputs.items():
        threshold = choose_threshold(data.y_val, output.val_probs, config.threshold)
        results[model_name] = compute_metrics(data.y_eval, output.eval_probs, threshold)
        scores = results[model_name].scores()
        logger.info(
            "test %-22s " + " | ".join(f"{k} {v:.3f}" for k, v in scores.items()), model_name
        )

    run_dir = Path(config.output_dir) / config.name
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    primary = config.models[0]
    figures = {
        "roc_curves": plots.roc_curves(
            data.y_eval, {name: output.eval_probs for name, output in outputs.items()}
        ),
        "confusion_matrix": plots.confusion(
            results[primary].confusion_matrix, f"Confusion matrix: {primary} (test set)"
        ),
    }
    if histories:
        figures["loss_curves"] = plots.loss_curves(histories)
    for name, fig in figures.items():
        fig.savefig(figures_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    if pairplot:
        shared = Path(config.output_dir) / "figures"
        shared.mkdir(parents=True, exist_ok=True)
        plots.pairplot(df).savefig(shared / "pairplot.png", dpi=100, bbox_inches="tight")
    logger.info("Saved figures to %s", figures_dir)

    pd.DataFrame(
        [{"model": name, **metrics.scores()} for name, metrics in results.items()]
    ).to_csv(run_dir / "test_results.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(config),
                "best_epochs": [history.best_epoch for history in histories],
                "test": {name: metrics.as_dict() for name, metrics in results.items()},
            },
            f,
            indent=2,
        )

    _append_trial(
        Path(config.output_dir) / "trials.csv",
        {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": config.name,
            "model": primary,
            "seed": config.seed,
            "hidden_sizes": "-".join(map(str, config.hidden_sizes)),
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "ensemble_size": config.ensemble_size,
            **{k: round(v, 4) for k, v in results[primary].scores().items()},
        },
    )
    return {name: metrics.as_dict() for name, metrics in results.items()}


def _update_summary(path: Path, summary: pd.DataFrame, config_name: str) -> None:
    """Replace this config's rows in the shared CV summary."""
    if path.exists():
        existing = pd.read_csv(path)
        summary = pd.concat([existing[existing["config"] != config_name], summary])
    summary.sort_values("roc_auc_mean", ascending=False).to_csv(path, index=False)


def _append_trial(path: Path, trial: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial))
        if is_new:
            writer.writeheader()
        writer.writerow(trial)
