import torch
from torch import nn

from diabetes_classifier.model import DiabetesClassifier


def test_forward_returns_one_logit_per_sample():
    model = DiabetesClassifier(n_features=8, hidden_sizes=[8, 16, 8])
    assert model(torch.randn(5, 8)).shape == (5,)


def test_dropout_follows_selected_layers():
    model = DiabetesClassifier(8, [8, 16, 32, 16, 8], dropout=0.5, dropout_layers=[3])
    layers = list(model.network)
    dropout_positions = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Dropout)]
    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]

    assert len(linear_layers) == 6  # five hidden layers + output
    assert dropout_positions == [8]  # after the ReLU of hidden layer index 3
