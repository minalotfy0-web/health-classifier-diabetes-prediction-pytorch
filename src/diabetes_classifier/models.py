"""Neural-network ensemble, classical baselines and their combination."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from diabetes_classifier.config import Config
from diabetes_classifier.data import FoldData
from diabetes_classifier.model import DiabetesClassifier
from diabetes_classifier.train import TrainingHistory, predict_proba, train_model

COMBINATIONS = {"nn+gradient_boosting": ("nn", "gradient_boosting")}


@dataclass
class ModelOutput:
    """Predicted probabilities on the validation and evaluation sets."""

    val_probs: np.ndarray
    eval_probs: np.ndarray


@dataclass
class FittedModels:
    """Trained networks (the "nn" ensemble) and scikit-learn models."""

    networks: list[DiabetesClassifier] = field(default_factory=list)
    sklearn: dict[str, ClassifierMixin] = field(default_factory=dict)
    histories: list[TrainingHistory] = field(default_factory=list)

    def predict_proba(self, name: str, X: np.ndarray) -> np.ndarray:
        """Probability of diabetes from model `name` (ensembles average their parts)."""
        if name == "nn":
            return np.mean([predict_proba(network, X) for network in self.networks], axis=0)
        if name in COMBINATIONS:
            return np.mean([self.predict_proba(part, X) for part in COMBINATIONS[name]], axis=0)
        return self.sklearn[name].predict_proba(X)[:, 1]

    def only(self, name: str) -> "FittedModels":
        """The models needed to predict with `name`, without training histories."""
        parts = set(COMBINATIONS.get(name, (name,)))
        return FittedModels(
            networks=self.networks if "nn" in parts else [],
            sklearn={key: model for key, model in self.sklearn.items() if key in parts},
        )


def fit_models(config: Config, data: FoldData, seed: int) -> FittedModels:
    """Train every model in `config.models`, plus the parts of any combination."""
    needed = set(config.models)
    for name in config.models:
        needed.update(COMBINATIONS.get(name, ()))

    fitted = FittedModels()
    if "nn" in needed:
        for member in range(config.ensemble_size):
            network, history = train_model(config, data, seed + member)
            fitted.networks.append(network)
            fitted.histories.append(history)
    for name in sorted(needed - {"nn"} - set(COMBINATIONS)):
        fitted.sklearn[name] = _sklearn_model(name, config, seed).fit(data.X_train, data.y_train)
    return fitted


def fit_predict(
    config: Config, data: FoldData, seed: int
) -> tuple[dict[str, ModelOutput], FittedModels]:
    """Train every model in `config.models` and predict the validation and evaluation sets."""
    fitted = fit_models(config, data, seed)
    outputs = {
        name: ModelOutput(
            fitted.predict_proba(name, data.X_val), fitted.predict_proba(name, data.X_eval)
        )
        for name in config.models
    }
    return outputs, fitted


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
