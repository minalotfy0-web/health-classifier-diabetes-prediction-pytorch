import dataclasses

import numpy as np
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from diabetes_classifier.data import cv_folds, split_development_test


@pytest.fixture
def tagged(df, small_config):
    """Development rows whose Age is a unique row id, plus a config that leaves it unchanged."""
    development, _ = split_development_test(df, small_config)
    development = development.assign(Age=np.arange(len(development), dtype=float))
    config = dataclasses.replace(
        small_config, zero_as_missing=[], missing_indicators=False, log_transform=[], cv_folds=4
    )
    return development, config


def evaluated_rows(fold) -> list[int]:
    """Row ids (the Age tag) of the fold's evaluation set, recovered from the scaled features."""
    column = fold.feature_names.index("Age")
    scaler = fold.preprocessor.scaler_
    ids = fold.X_eval[:, column] * scaler.scale_[column] + scaler.mean_[column]
    return np.rint(ids).astype(int).tolist()


def test_each_fold_evaluates_a_disjoint_part_covering_the_development_set(tagged):
    development, config = tagged
    splitter = StratifiedKFold(config.cv_folds, shuffle=True, random_state=config.seed)
    folds = list(cv_folds(development, splitter, config))

    assert len(folds) == config.cv_folds
    seen = [row for fold in folds for row in evaluated_rows(fold)]
    assert len(seen) == len(set(seen)) == len(development)
    assert sorted(seen) == list(range(len(development)))


def test_fold_parts_have_the_expected_sizes(tagged):
    development, config = tagged
    splitter = StratifiedKFold(config.cv_folds, shuffle=True, random_state=config.seed)
    for fold in cv_folds(development, splitter, config):
        total = len(fold.y_train) + len(fold.y_val) + len(fold.y_eval)
        assert total == len(development)
        assert abs(len(fold.y_eval) - len(development) / config.cv_folds) <= 1
        assert len(fold.y_val) > 0


def test_repeated_cv_gives_folds_times_repeats_and_covers_each_repeat(tagged):
    development, config = tagged
    config = dataclasses.replace(config, cv_folds=3, cv_repeats=2)
    splitter = RepeatedStratifiedKFold(
        n_splits=config.cv_folds, n_repeats=config.cv_repeats, random_state=config.seed
    )
    folds = list(cv_folds(development, splitter, config))

    assert len(folds) == config.cv_folds * config.cv_repeats
    for repeat in range(config.cv_repeats):
        part = folds[repeat * config.cv_folds : (repeat + 1) * config.cv_folds]
        seen = sorted(row for fold in part for row in evaluated_rows(fold))
        assert seen == list(range(len(development)))


def test_fold_preprocessing_is_fitted_on_that_folds_training_rows(tagged):
    development, config = tagged
    splitter = StratifiedKFold(config.cv_folds, shuffle=True, random_state=config.seed)
    for fold in cv_folds(development, splitter, config):
        assert np.abs(fold.X_train.mean(axis=0)).max() < 1e-4
