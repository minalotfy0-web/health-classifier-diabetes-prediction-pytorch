import torch
from torch import nn

from diabetes_classifier.model import DiabetesClassifier


def test_forward_returns_one_logit_per_sample():
    model = DiabetesClassifier(n_features=13, hidden_sizes=[32, 16])
    assert model(torch.randn(5, 13)).shape == (5,)


def test_dropout_follows_selected_layers():
    model = DiabetesClassifier(8, [8, 16, 32, 16, 8], dropout=0.5, dropout_layers=[3])
    layers = list(model.network)
    dropout_positions = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Dropout)]
    assert dropout_positions == [8]  # after the ReLU of hidden layer index 3


def test_dropout_after_every_layer_by_default():
    model = DiabetesClassifier(8, [32, 16, 8], dropout=0.2)
    assert sum(isinstance(layer, nn.Dropout) for layer in model.network) == 3


def test_batch_norm_layers():
    model = DiabetesClassifier(8, [32, 16], batch_norm=True)
    assert sum(isinstance(layer, nn.BatchNorm1d) for layer in model.network) == 2
