"""Neural-network training: mini-batches, early stopping and best-epoch restoration."""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from diabetes_classifier.config import Config
from diabetes_classifier.data import FoldData
from diabetes_classifier.model import DiabetesClassifier

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Per-epoch losses and the epoch with the lowest validation loss."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.inf


def train_model(
    config: Config, data: FoldData, seed: int
) -> tuple[DiabetesClassifier, TrainingHistory]:
    """Train one network and restore the weights from its best validation epoch."""
    torch.manual_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)
    X_train, y_train = torch.from_numpy(data.X_train), torch.from_numpy(data.y_train)
    X_val, y_val = torch.from_numpy(data.X_val), torch.from_numpy(data.y_val)

    model = DiabetesClassifier(
        data.n_features,
        config.hidden_sizes,
        config.dropout,
        config.dropout_layers,
        config.batch_norm,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(config.pos_weight, y_train))
    optimizer = _make_optimizer(config, model)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=max(1, (config.patience or 40) // 4)
        )
        if config.lr_scheduler
        else None
    )
    batch_size = config.batch_size or len(y_train)

    history = TrainingHistory()
    best_state = _snapshot(model)
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in torch.randperm(len(y_train), generator=shuffle_generator).split(batch_size):
            if config.batch_norm and len(batch) < 2:
                continue  # batch norm needs at least two samples
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            seen += len(batch)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if scheduler is not None:
            scheduler.step(val_loss)

        history.train_loss.append(total_loss / seen)
        history.val_loss.append(val_loss)
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = _snapshot(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % config.log_every == 0:
            logger.debug(
                "epoch %5d | train loss %.4f | val loss %.4f", epoch, total_loss / seen, val_loss
            )
        if config.patience is not None and epochs_without_improvement >= config.patience:
            logger.debug("Early stop at epoch %d", epoch)
            break

    model.load_state_dict(best_state)
    return model, history


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Predicted probability of diabetes for each sample."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def _pos_weight(setting: float | str | None, y_train: torch.Tensor) -> torch.Tensor | None:
    if setting is None:
        return None
    if setting == "auto":
        positives = y_train.sum()
        if positives == 0:
            raise ValueError("pos_weight='auto' needs at least one positive training sample")
        return (len(y_train) - positives) / positives
    return torch.tensor(float(setting))


def _make_optimizer(config: Config, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
