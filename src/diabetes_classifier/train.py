"""Training loop with best-epoch checkpointing on the validation set."""

import logging
import math
from dataclasses import dataclass, field

import torch
from torch import nn

from diabetes_classifier.data import DataSplits

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Per-epoch losses and the epoch with the lowest validation loss."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.inf


def train(
    model: nn.Module,
    splits: DataSplits,
    epochs: int,
    learning_rate: float,
    log_every: int = 500,
) -> TrainingHistory:
    """Train with full-batch SGD and restore the weights from the best validation epoch."""
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = TrainingHistory()
    best_state = _snapshot(model)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(splits.X_train), splits.y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(splits.X_val), splits.y_val).item()

        history.train_loss.append(train_loss.item())
        history.val_loss.append(val_loss)
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = _snapshot(model)

        if epoch == 1 or epoch % log_every == 0:
            logger.info(
                "epoch %5d | train loss %.4f | val loss %.4f",
                epoch,
                train_loss.item(),
                val_loss,
            )

    model.load_state_dict(best_state)
    logger.info(
        "Best validation loss %.4f at epoch %d", history.best_val_loss, history.best_epoch
    )
    return history


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
