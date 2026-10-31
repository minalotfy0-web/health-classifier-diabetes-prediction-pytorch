"""Fully connected binary classifier."""

from collections.abc import Sequence

import torch
from torch import nn


class DiabetesClassifier(nn.Module):
    """Feed-forward network with ReLU hidden layers and a single logit output.

    Args:
        n_features: Number of input features.
        hidden_sizes: Width of each hidden layer, in order.
        dropout: Dropout probability.
        dropout_layers: Indices of the hidden layers that are followed by dropout.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Sequence[int],
        dropout: float = 0.5,
        dropout_layers: Sequence[int] = (),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = n_features
        for index, width in enumerate(hidden_sizes):
            layers += [nn.Linear(in_features, width), nn.ReLU()]
            if index in dropout_layers:
                layers.append(nn.Dropout(dropout))
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per sample (apply a sigmoid for probabilities)."""
        return self.network(x).squeeze(-1)
