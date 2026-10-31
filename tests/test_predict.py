"""The saved Predictor: the committed final model, malformed input and persistence."""

import dataclasses
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from diabetes_classifier.config import load_config
from diabetes_classifier.data import FEATURE_COLUMNS, TARGET_COLUMN, split_development_test
from diabetes_classifier.evaluate import compute_metrics
from diabetes_classifier.predict import Predictor

TUNED_DIR = Path(__file__).resolve().parents[1] / "results" / "tuned"


def test_tuned_model_loads(tuned_predictor):
    assert tuned_predictor.model_name == "nn+gradient_boosting"
    assert 0 < tuned_predictor.threshold < 1
    assert len(tuned_predictor.models.networks) > 0
    assert "gradient_boosting" in tuned_predictor.models.sklearn


def test_tuned_model_reproduces_the_reported_test_scores(tuned_predictor, df):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "tuned.yaml")
    _, test = split_development_test(df, config)
    probabilities = tuned_predictor.predict_proba(test.drop(columns=TARGET_COLUMN))
    metrics = compute_metrics(test[TARGET_COLUMN], probabilities, tuned_predictor.threshold)

    reported = pd.read_csv(TUNED_DIR / "test_results.csv").set_index("model")
    row = reported.loc[tuned_predictor.model_name]
    assert metrics.accuracy == pytest.approx(row["accuracy"], abs=0.02)
    assert metrics.roc_auc == pytest.approx(row["roc_auc"], abs=0.01)


def test_probabilities_are_valid(tuned_predictor, patients):
    probabilities = tuned_predictor.predict_proba(patients)
    assert probabilities.shape == (len(patients),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_predict_applies_the_threshold(tuned_predictor, patients):
    probabilities = tuned_predictor.predict_proba(patients)
    predictions = tuned_predictor.predict(patients)
    assert set(predictions) <= {0, 1}
    np.testing.assert_array_equal(predictions, (probabilities >= tuned_predictor.threshold) * 1)


def test_a_patient_scores_the_same_alone_or_in_a_batch(tuned_predictor, patients):
    batch = tuned_predictor.predict_proba(patients)
    for index in (0, 7, 19):
        alone = tuned_predictor.predict_proba(patients.iloc[[index]])
        np.testing.assert_allclose(alone, batch[[index]], atol=1e-5)


def test_extra_columns_and_column_order_are_ignored(tuned_predictor, patients):
    expected = tuned_predictor.predict_proba(patients)
    shuffled = patients[list(reversed(patients.columns))].assign(Outcome=1, Note="x")
    np.testing.assert_allclose(tuned_predictor.predict_proba(shuffled), expected, atol=1e-6)


def test_threshold_zero_predicts_everyone_positive(tuned_predictor, patients):
    predictor = dataclasses.replace(tuned_predictor, threshold=0.0)
    assert predictor.predict(patients).tolist() == [1] * len(patients)


def test_threshold_above_one_predicts_everyone_negative(tuned_predictor, patients):
    predictor = dataclasses.replace(tuned_predictor, threshold=1.5)
    assert predictor.predict(patients).tolist() == [0] * len(patients)


def test_a_probability_equal_to_the_threshold_is_positive(tuned_predictor, patients):
    probabilities = tuned_predictor.predict_proba(patients)
    predictor = dataclasses.replace(tuned_predictor, threshold=float(probabilities[3]))
    predictions = predictor.predict(patients)
    assert predictions[3] == 1
    np.testing.assert_array_equal(predictions, (probabilities >= probabilities[3]) * 1)


@pytest.mark.parametrize("column", FEATURE_COLUMNS)
def test_missing_column_raises(tuned_predictor, patients, column):
    with pytest.raises(ValueError):
        tuned_predictor.predict_proba(patients.drop(columns=column))
    with pytest.raises(ValueError):
        tuned_predictor.predict(patients.drop(columns=column))


def test_no_columns_at_all_raises(tuned_predictor):
    with pytest.raises(ValueError):
        tuned_predictor.predict_proba(pd.DataFrame({"Outcome": [1]}))


def test_text_values_raise(tuned_predictor, patients):
    bad = patients.astype({"Glucose": object})
    bad.loc[0, "Glucose"] = "high"
    with pytest.raises(ValueError):
        tuned_predictor.predict_proba(bad)


def test_zero_means_not_measured(tuned_predictor, patients):
    zeros = patients.assign(Glucose=0, BloodPressure=0, SkinThickness=0, Insulin=0, BMI=0)
    probabilities = tuned_predictor.predict_proba(zeros)
    assert np.isfinite(probabilities).all()


def test_zero_pregnancies_is_a_real_value(tuned_predictor, patients):
    # Only the five lab/body columns are imputed; Pregnancies=0 must not crash or become NaN
    probabilities = tuned_predictor.predict_proba(patients.assign(Pregnancies=0))
    assert np.isfinite(probabilities).all()


@pytest.mark.parametrize("missing", [np.nan, None], ids=["nan", "none"])
@pytest.mark.parametrize("column", ["Glucose", "BMI", "Age", "Pregnancies"])
def test_missing_values_raise(tuned_predictor, patients, column, missing):
    row = patients.iloc[[0]].astype({column: object})
    row.loc[row.index[0], column] = missing
    with pytest.raises(ValueError):
        tuned_predictor.predict_proba(row)
    with pytest.raises(ValueError):
        tuned_predictor.predict(row)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, -1.0])
def test_infinite_and_negative_values_raise(tuned_predictor, patients, bad):
    row = patients.iloc[[0]].astype({"Glucose": float})
    row["Glucose"] = bad
    with pytest.raises(ValueError):
        tuned_predictor.predict_proba(row)


def test_empty_frame_raises(tuned_predictor, patients):
    with pytest.raises(ValueError):
        tuned_predictor.predict_proba(patients.iloc[0:0])


def test_save_and_load_round_trip(tuned_predictor, patients, tmp_path):
    path = tmp_path / "copy.joblib"
    tuned_predictor.save(path)
    loaded = Predictor.load(path)
    assert loaded.model_name == tuned_predictor.model_name
    assert loaded.threshold == tuned_predictor.threshold
    np.testing.assert_allclose(
        loaded.predict_proba(patients), tuned_predictor.predict_proba(patients), atol=1e-6
    )


def test_loading_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor.load(tmp_path / "nope.joblib")


def test_loading_something_else_raises(tmp_path):
    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a predictor"}, path)
    with pytest.raises(TypeError):
        Predictor.load(path)


def test_loading_a_corrupt_file_raises(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"this is not a joblib file")
    with pytest.raises(Exception):  # noqa: B017 - the exact type depends on the joblib version
        Predictor.load(path)
