"""Decision thresholds and classification metrics."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


@dataclass
class Metrics:
    """Scores of one model at one decision threshold."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def scores(self) -> dict[str, float]:
        """The scalar metrics only (no threshold or confusion matrix)."""
        return {name: getattr(self, name) for name in METRIC_NAMES}


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, method: float | str) -> float:
    """Pick the decision threshold on validation data.

    `method` is a fixed probability, "f1" (maximise F1) or "youden" (maximise
    sensitivity + specificity - 1).
    """
    if not isinstance(method, str):
        return float(method)
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        return float(min(thresholds[np.argmax(tpr - fpr)], 1.0))
    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        return float(thresholds[np.argmax(f1[:-1])])
    raise ValueError(f"Unknown threshold method {method!r}")


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Metrics:
    """Accuracy, precision, recall, F1, ROC AUC and the confusion matrix at `threshold`."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (probabilities >= threshold).astype(int)
    return Metrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    )
