import torch

from diabetes_classifier.data import load_dataset


def test_dataset_shape(data_path):
    df = load_dataset(data_path)
    assert df.shape == (768, 9)
    assert set(df["Outcome"].unique()) == {0, 1}


def test_split_sizes(splits):
    assert (len(splits.y_train), len(splits.y_val), len(splits.y_test)) == (537, 77, 154)
    assert splits.n_features == 8


def test_training_features_are_standardised(splits):
    assert torch.allclose(splits.X_train.mean(dim=0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(splits.X_train.std(dim=0, unbiased=False), torch.ones(8), atol=1e-4)


def test_splits_are_stratified(splits):
    rates = [y.mean().item() for y in (splits.y_train, splits.y_val, splits.y_test)]
    assert max(rates) - min(rates) < 0.02
