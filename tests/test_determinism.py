"""The same seed must give the same splits, predictions and metrics."""

import dataclasses
import random

import numpy as np
import pandas as pd
import pytest
import torch

from diabetes_classifier.cli import main
from diabetes_classifier.config import MODEL_NAMES, save_config
from diabetes_classifier.data import stratified_split
from diabetes_classifier.experiment import run_cv, run_final, set_seed
from diabetes_classifier.models import fit_predict
from diabetes_classifier.predict import Predictor
from diabetes_classifier.train import predict_proba, train_model


@pytest.fixture
def tiny_config(small_config):
    return dataclasses.replace(
        small_config, models=["nn", "logistic_regression"], ensemble_size=1, max_epochs=5
    )


def in_folder(config, folder):
    return dataclasses.replace(config, output_dir=str(folder))


def test_set_seed_makes_every_generator_repeatable():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()

    set_seed(11)
    first = draw()
    set_seed(11)
    assert draw() == first
    set_seed(12)
    assert draw() != first


def test_split_is_repeatable_and_depends_on_the_seed(df):
    _, a = stratified_split(df, 0.2, seed=3)
    _, b = stratified_split(df, 0.2, seed=3)
    _, c = stratified_split(df, 0.2, seed=4)
    assert a.index.tolist() == b.index.tolist()
    assert a.index.tolist() != c.index.tolist()


def test_a_network_trains_identically_twice(small_config, fold):
    config = dataclasses.replace(small_config, max_epochs=10, patience=None)
    first, first_history = train_model(config, fold, seed=5)
    second, second_history = train_model(config, fold, seed=5)

    assert first_history.train_loss == pytest.approx(second_history.train_loss, abs=1e-6)
    assert first_history.val_loss == pytest.approx(second_history.val_loss, abs=1e-6)
    np.testing.assert_allclose(
        predict_proba(first, fold.X_eval), predict_proba(second, fold.X_eval), atol=1e-6
    )


def test_a_different_seed_gives_a_different_network(small_config, fold):
    config = dataclasses.replace(small_config, max_epochs=10, patience=None)
    first, _ = train_model(config, fold, seed=5)
    other, _ = train_model(config, fold, seed=6)
    assert not np.allclose(predict_proba(first, fold.X_eval), predict_proba(other, fold.X_eval))


def test_every_model_predicts_identically_twice(small_config, fold):
    config = dataclasses.replace(small_config, max_epochs=5, patience=None)
    first, _ = fit_predict(config, fold, seed=2)
    second, _ = fit_predict(config, fold, seed=2)

    assert list(first) == list(second) == list(MODEL_NAMES)
    for name in MODEL_NAMES:
        np.testing.assert_allclose(first[name].val_probs, second[name].val_probs, atol=1e-6)
        np.testing.assert_allclose(first[name].eval_probs, second[name].eval_probs, atol=1e-6)


def test_run_final_is_repeatable(tiny_config, df, tmp_path):
    first = run_final(in_folder(tiny_config, tmp_path / "a"))
    second = run_final(in_folder(tiny_config, tmp_path / "b"))

    assert first.keys() == second.keys()
    for name in first:
        assert first[name]["confusion_matrix"] == second[name]["confusion_matrix"]
        assert first[name]["threshold"] == pytest.approx(second[name]["threshold"], abs=1e-6)
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert first[name][metric] == pytest.approx(second[name][metric], abs=1e-6)

    patients = df.head(15).drop(columns="Outcome")
    saved_a = Predictor.load(tmp_path / "a" / "default" / "model.joblib")
    saved_b = Predictor.load(tmp_path / "b" / "default" / "model.joblib")
    np.testing.assert_allclose(
        saved_a.predict_proba(patients), saved_b.predict_proba(patients), atol=1e-6
    )
    assert saved_a.predict(patients).tolist() == saved_b.predict(patients).tolist()


def test_run_final_depends_on_the_seed(tiny_config, tmp_path):
    first = run_final(in_folder(tiny_config, tmp_path / "a"))
    other = run_final(in_folder(dataclasses.replace(tiny_config, seed=1), tmp_path / "b"))
    assert first["nn"] != other["nn"]


def test_run_cv_is_repeatable(tiny_config, tmp_path):
    first = run_cv(in_folder(tiny_config, tmp_path / "a"))
    second = run_cv(in_folder(tiny_config, tmp_path / "b"))
    pd.testing.assert_frame_equal(first, second, atol=1e-6)

    folds_a = pd.read_csv(tmp_path / "a" / "cv" / "default_folds.csv")
    folds_b = pd.read_csv(tmp_path / "b" / "cv" / "default_folds.csv")
    pd.testing.assert_frame_equal(folds_a, folds_b, atol=1e-6)


def test_cli_train_is_repeatable(tiny_config, tmp_path):
    config_path = tmp_path / "tiny.yaml"
    save_config(tiny_config, config_path)
    for folder in ("a", "b"):
        main(["train", "--config", str(config_path), "--output-dir", str(tmp_path / folder)])

    results = [pd.read_csv(tmp_path / folder / "default" / "test_results.csv") for folder in "ab"]
    pd.testing.assert_frame_equal(results[0], results[1], atol=1e-6)
