"""Figures for an experiment run. Each function returns the figure; the caller saves it."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from diabetes_classifier.data import TARGET_COLUMN
from diabetes_classifier.train import TrainingHistory


def loss_curves(history: TrainingHistory) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.train_loss, label="Training loss", color="tab:blue", linewidth=1)
    ax.plot(history.val_loss, label="Validation loss", color="tab:red", linewidth=1.5)
    ax.axvline(
        history.best_epoch - 1,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch ({history.best_epoch})",
    )
    ax.set(xlabel="Epoch", ylabel="BCE loss", title="Training and validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def roc(y_true: np.ndarray, probabilities: np.ndarray, auc: float) -> Figure:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="tab:blue", label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def confusion(matrix: list[list[int]]) -> Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    labels = ["No diabetes", "Diabetes"]
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)")
    fig.tight_layout()
    return fig


def pairplot(df: pd.DataFrame) -> Figure:
    """Pairwise feature scatter plots coloured by outcome (slow: ~1 min)."""
    grid = sns.pairplot(df, hue=TARGET_COLUMN, diag_kind="kde", palette="viridis", height=2)
    return grid.figure
