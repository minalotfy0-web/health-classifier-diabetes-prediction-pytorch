"""Hyperparameter search for the network with Optuna, scored by cross-validated ROC AUC."""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from diabetes_classifier.config import Config
from diabetes_classifier.data import (
    TARGET_COLUMN,
    FoldData,
    load_dataset,
    prepare_fold,
    split_development_test,
    stratified_split,
)
from diabetes_classifier.train import predict_proba, train_model

logger = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial) -> dict:
    """Search space: depth, width (halving per layer), regularisation and optimiser."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    width = trial.suggest_categorical("width", [16, 32, 64, 128])
    return {
        "hidden_sizes": [max(8, width // 2**i) for i in range(n_layers)],
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "batch_norm": trial.suggest_categorical("batch_norm", [False, True]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


def tune(
    config: Config, n_trials: int, timeout: float | None = None
) -> tuple[Config, optuna.Study]:
    """Search network hyperparameters on the development set (the test set is not used).

    Each trial trains a single network per fold of `config.cv_folds`-fold CV and is scored
    by mean ROC AUC; unpromising trials are pruned after each fold. Returns `config`
    updated with the best parameters (named "tuned") and the study.
    """
    folds = _prepare_folds(config)

    def objective(trial: optuna.Trial) -> float:
        trial_config = dataclasses.replace(
            config, **suggest_params(trial), dropout_layers=None, ensemble_size=1
        )
        aucs = []
        for index, data in enumerate(folds):
            model, _ = train_model(trial_config, data, seed=config.seed + index)
            aucs.append(roc_auc_score(data.y_eval, predict_proba(model, data.X_eval)))
            trial.report(float(np.mean(aucs)), index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.info("Best CV ROC AUC %.4f with %s", study.best_value, study.best_params)

    best = optuna.trial.FixedTrial(study.best_params)
    tuned = dataclasses.replace(config, name="tuned", **suggest_params(best), dropout_layers=None)
    return tuned, study


def save_study(study: optuna.Study, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "tuning" / "trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(path, index=False)
    return path


def _prepare_folds(config: Config) -> list[FoldData]:
    development, _ = split_development_test(load_dataset(config.data_path), config)
    splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    folds = []
    for index, (fit_index, eval_index) in enumerate(
        splitter.split(development, development[TARGET_COLUMN])
    ):
        train_df, val_df = stratified_split(
            development.iloc[fit_index], config.val_fraction_of_development, config.seed + index
        )
        folds.append(prepare_fold(train_df, val_df, development.iloc[eval_index], config))
    return folds
