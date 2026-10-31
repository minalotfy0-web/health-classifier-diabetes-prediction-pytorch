"""Figures. Each function returns the figure; the caller saves it."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve

from diabetes_classifier.data import TARGET_COLUMN
from diabetes_classifier.train import TrainingHistory


def loss_curves(histories: list[TrainingHistory]) -> Figure:
    """Training and validation loss of each network in the ensemble."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, history in enumerate(histories):
        epochs = np.arange(1, len(history.val_loss) + 1)
        first = index == 0
        ax.plot(
            epochs,
            history.train_loss,
            color="tab:blue",
            alpha=0.35,
            linewidth=1,
            label="Training loss" if first else None,
        )
        ax.plot(
            epochs,
            history.val_loss,
            color="tab:red",
            linewidth=1.5,
            label="Validation loss" if first else None,
        )
        ax.axvline(
            history.best_epoch,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            label="Best epoch" if first else None,
        )
    title = "Training and validation loss"
    if len(histories) > 1:
        title += f" ({len(histories)} ensemble members)"
    ax.set(xlabel="Epoch", ylabel="BCE loss", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def roc_curves(y_true: np.ndarray, probabilities: dict[str, np.ndarray]) -> Figure:
    """ROC curve of each model on the test set."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, probs):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def confusion(matrix: list[list[int]], title: str) -> Figure:
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
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    return fig


def cv_comparison(folds: pd.DataFrame) -> Figure:
    """Distribution of fold scores per configuration and model."""
    folds = folds.assign(setup=folds["config"] + " / " + folds["model"])
    order = folds.groupby("setup")["roc_auc"].mean().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(order) + 2), sharey=True)
    for ax, metric in zip(axes, ("roc_auc", "accuracy"), strict=True):
        sns.boxplot(data=folds, x=metric, y="setup", order=order, color="tab:blue", ax=ax)
        ax.set(xlabel=metric.replace("_", " ").upper() if metric == "roc_auc" else "Accuracy")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Repeated stratified cross-validation (development set)")
    fig.tight_layout()
    return fig


def pairplot(df: pd.DataFrame) -> Figure:
    """Pairwise feature scatter plots coloured by outcome (slow: ~1 min)."""
    grid = sns.pairplot(df, hue=TARGET_COLUMN, diag_kind="kde", palette="viridis", height=2)
    return grid.figure
