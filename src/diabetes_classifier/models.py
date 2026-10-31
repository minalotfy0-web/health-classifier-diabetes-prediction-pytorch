"""Neural-network ensemble, classical baselines and their combination."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from diabetes_classifier.config import Config
from diabetes_classifier.data import FoldData
from diabetes_classifier.train import TrainingHistory, predict_proba, train_model

COMBINATIONS = {"nn+gradient_boosting": ("nn", "gradient_boosting")}


@dataclass
class ModelOutput:
    """Predicted probabilities on the validation and evaluation sets."""

    val_probs: np.ndarray
    eval_probs: np.ndarray


def fit_predict(
    config: Config, data: FoldData, seed: int
) -> tuple[dict[str, ModelOutput], list[TrainingHistory]]:
    """Train every model in `config.models` and return its predictions.

    Also returns the training histories of the networks in the ensemble.
    """
    needed = set(config.models)
    for name in config.models:
        needed.update(COMBINATIONS.get(name, ()))

    outputs: dict[str, ModelOutput] = {}
    histories: list[TrainingHistory] = []
    if "nn" in needed:
        outputs["nn"], histories = _fit_nn_ensemble(config, data, seed)
    for name in sorted(needed - {"nn"} - set(COMBINATIONS)):
        model = _sklearn_model(name, config, seed).fit(data.X_train, data.y_train)
        outputs[name] = ModelOutput(
            model.predict_proba(data.X_val)[:, 1], model.predict_proba(data.X_eval)[:, 1]
        )
    for name, parts in COMBINATIONS.items():
        if name in needed:
            outputs[name] = ModelOutput(
                np.mean([outputs[p].val_probs for p in parts], axis=0),
                np.mean([outputs[p].eval_probs for p in parts], axis=0),
            )
    return {name: outputs[name] for name in config.models}, histories


def _fit_nn_ensemble(
    config: Config, data: FoldData, seed: int
) -> tuple[ModelOutput, list[TrainingHistory]]:
    """Average the probabilities of `ensemble_size` networks trained with different seeds."""
    val_probs, eval_probs, histories = [], [], []
    for member in range(config.ensemble_size):
        model, history = train_model(config, data, seed + member)
        val_probs.append(predict_proba(model, data.X_val))
        eval_probs.append(predict_proba(model, data.X_eval))
        histories.append(history)
    return ModelOutput(np.mean(val_probs, axis=0), np.mean(eval_probs, axis=0)), histories


def _sklearn_model(name: str, config: Config, seed: int) -> ClassifierMixin:
    class_weight = "balanced" if config.pos_weight is not None else None
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=class_weight)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_depth=3,
            l2_regularization=1.0,
            class_weight=class_weight,
            random_state=seed,
        )
    raise ValueError(f"Unknown model {name!r}")
