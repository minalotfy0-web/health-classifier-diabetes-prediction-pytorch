"""Nothing learned from validation/test rows may influence preprocessing or training data."""

import numpy as np
import pandas as pd
import pytest

from diabetes_classifier.data import (
    TARGET_COLUMN,
    prepare_fold,
    split_development_test,
    stratified_split,
)
from diabetes_classifier.preprocessing import Preprocessor


def make_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Glucose": [0.0, 90.0, 100.0, 110.0, 120.0, 130.0],
            "Insulin": [0.0, 0.0, 50.0, 60.0, 70.0, 80.0],
            "Age": [21.0, 25.0, 30.0, 35.0, 40.0, 60.0],
        }
    )


def make_other() -> pd.DataFrame:
    """Rows from a very different distribution."""
    return pd.DataFrame(
        {
            "Glucose": [0.0, 400.0, 500.0],
            "Insulin": [0.0, 900.0, 0.0],
            "Age": [80.0, 81.0, 90.0],
        }
    )


def make_preprocessor() -> Preprocessor:
    return Preprocessor(["Glucose", "Insulin"], missing_indicators=True, log_transform=["Age"])


def test_transform_does_not_change_the_fitted_statistics():
    pre = make_preprocessor().fit(make_train())
    medians = dict(pre.medians_)
    mean, scale = pre.scaler_.mean_.copy(), pre.scaler_.scale_.copy()
    before = pre.transform(make_train())

    pre.transform(make_other())

    assert pre.medians_ == medians
    np.testing.assert_array_equal(pre.scaler_.mean_, mean)
    np.testing.assert_array_equal(pre.scaler_.scale_, scale)
    np.testing.assert_array_equal(pre.transform(make_train()), before)


def test_fitted_statistics_depend_only_on_the_training_rows():
    pre = make_preprocessor().fit(make_train())
    assert pre.medians_ == {"Glucose": 110.0, "Insulin": 65.0}

    # Fitting on the training rows again, after having seen other data, gives the same result
    other = make_preprocessor().fit(make_other())
    again = make_preprocessor().fit(make_train())
    assert other.medians_ != pre.medians_
    np.testing.assert_array_equal(again.scaler_.mean_, pre.scaler_.mean_)
    np.testing.assert_array_equal(again.transform(make_other()), pre.transform(make_other()))


def test_unseen_rows_are_scaled_with_training_statistics():
    pre = make_preprocessor().fit(make_train())
    other = make_other()
    expected = (pre._transform_frame(other) - pre.scaler_.mean_) / pre.scaler_.scale_
    np.testing.assert_allclose(pre.transform(other), expected.to_numpy(), rtol=1e-5, atol=1e-5)
    # The shifted rows are not standardised on their own
    assert np.abs(pre.transform(other).mean(axis=0)).max() > 1


def test_transform_does_not_modify_its_input():
    other = make_other()
    snapshot = other.copy()
    make_preprocessor().fit(make_train()).transform(other)
    pd.testing.assert_frame_equal(other, snapshot)


def test_columns_not_listed_are_left_alone():
    train = make_train().assign(Pregnancies=[0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
    pre = Preprocessor(["Glucose"]).fit(train)
    assert (pre._transform_frame(train)["Pregnancies"] == train["Pregnancies"]).all()


def test_transform_before_fit_raises():
    with pytest.raises(AttributeError):
        make_preprocessor().transform(make_train())


def test_transform_with_missing_column_raises():
    pre = make_preprocessor().fit(make_train())
    with pytest.raises(KeyError):
        pre.transform(make_train().drop(columns="Glucose"))


@pytest.mark.xfail(strict=True, reason="a column that is all zeros in training gets a NaN median")
def test_all_zero_training_column_gives_finite_features():
    train = make_train().assign(Insulin=0.0)
    X = Preprocessor(["Insulin"]).fit(train).transform(train)
    assert np.isfinite(X).all()


def split_parts(df, config):
    development, test = split_development_test(df, config)
    train, val = stratified_split(development, config.val_fraction_of_development, config.seed)
    return train, val, test


def test_split_parts_are_disjoint_and_complete(df, small_config):
    train, val, test = split_parts(df, small_config)
    indices = [set(part.index) for part in (train, val, test)]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert set().union(*indices) == set(df.index)


def test_fold_statistics_come_from_the_training_rows_only(df, small_config):
    train, val, test = split_parts(df, small_config)
    fold = prepare_fold(train, val, test, small_config)

    for column in small_config.zero_as_missing:
        nonzero = train.loc[train[column] != 0, column]
        assert fold.preprocessor.medians_[column] == pytest.approx(nonzero.median())

    features = train.drop(columns=TARGET_COLUMN)
    transformed = fold.preprocessor._transform_frame(features)
    np.testing.assert_allclose(fold.preprocessor.scaler_.mean_, transformed.mean(), rtol=1e-6)


def test_changing_validation_and_test_rows_does_not_change_training_features(df, small_config):
    train, val, test = split_parts(df, small_config)
    reference = prepare_fold(train, val, test, small_config)

    shifted_val = val.assign(Glucose=val["Glucose"] + 100, Age=val["Age"] + 30)
    shifted_test = test.assign(Insulin=test["Insulin"] * 5 + 7, BMI=test["BMI"] + 20)
    changed = prepare_fold(train, shifted_val, shifted_test, small_config)

    np.testing.assert_array_equal(changed.X_train, reference.X_train)
    np.testing.assert_array_equal(changed.y_train, reference.y_train)
    assert changed.preprocessor.medians_ == reference.preprocessor.medians_
    np.testing.assert_array_equal(
        changed.preprocessor.scaler_.mean_, reference.preprocessor.scaler_.mean_
    )
    np.testing.assert_array_equal(
        changed.preprocessor.scaler_.scale_, reference.preprocessor.scaler_.scale_
    )
    # ... while the held-out features do reflect the new values
    assert not np.array_equal(changed.X_val, reference.X_val)
    assert not np.array_equal(changed.X_eval, reference.X_eval)


def test_evaluation_features_are_not_standardised_on_themselves(fold):
    assert np.abs(fold.X_train.mean(axis=0)).max() < 1e-5
    assert np.abs(fold.X_eval.mean(axis=0)).max() > 1e-4
    assert np.abs(fold.X_val.mean(axis=0)).max() > 1e-4


def test_dropping_validation_rows_leaves_the_fit_unchanged(df, small_config):
    train, val, test = split_parts(df, small_config)
    full = prepare_fold(train, val, test, small_config)
    smaller = prepare_fold(train, val.head(10), test.head(10), small_config)
    np.testing.assert_array_equal(smaller.X_train, full.X_train)
    assert smaller.X_eval.shape[0] == 10
