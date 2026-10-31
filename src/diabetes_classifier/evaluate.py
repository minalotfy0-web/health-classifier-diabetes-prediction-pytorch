"""Classification metrics."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict:
        return asdict(self)


def predict_proba(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Predicted probability of diabetes for each sample."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def evaluate(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, threshold: float = 0.5
) -> Metrics:
    """Compute accuracy, precision, recall, F1, ROC AUC and the confusion matrix."""
    y_true = y.numpy().astype(int)
    probabilities = predict_proba(model, X)
    y_pred = (probabilities >= threshold).astype(int)
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    )
