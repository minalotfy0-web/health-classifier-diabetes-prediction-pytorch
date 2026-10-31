import numpy as np
import pytest

from diabetes_classifier.config import Config


def test_dataset_shape(df):
    assert df.shape == (768, 9)
    assert set(df["Outcome"].unique()) == {0, 1}


def test_split_sizes(fold):
    assert (len(fold.y_train), len(fold.y_val), len(fold.y_eval)) == (537, 77, 154)


def test_missing_indicators_are_added(fold):
    assert fold.n_features == 8 + 5
    assert "Insulin_missing" in fold.feature_names


def test_training_features_are_standardised(fold):
    informative = fold.X_train[:, fold.X_train.std(axis=0) > 0]
    np.testing.assert_allclose(informative.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(informative.std(axis=0), 1, atol=1e-4)


def test_splits_are_stratified(fold):
    rates = [y.mean() for y in (fold.y_train, fold.y_val, fold.y_eval)]
    assert max(rates) - min(rates) < 0.02


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        Config(threshold="median")
    with pytest.raises(ValueError):
        Config(models=["svm"])
