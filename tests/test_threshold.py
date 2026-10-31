"""Edge cases for threshold selection and the metrics computed from a threshold."""

import math

import numpy as np
import pytest

from diabetes_classifier.evaluate import choose_threshold, compute_metrics

y_true = np.array([0, 0, 0, 1, 1, 1])
probs = np.array([0.1, 0.2, 0.45, 0.4, 0.7, 0.9])


@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0, 0.25])
def test_fixed_thresholds_are_returned_as_floats(value):
    threshold = choose_threshold(y_true, probs, value)
    assert threshold == value
    assert isinstance(threshold, float)


def test_threshold_zero_predicts_everyone_positive():
    metrics = compute_metrics(y_true, probs, 0.0)
    assert metrics.confusion_matrix == [[0, 3], [0, 3]]
    assert metrics.recall == 1.0
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.accuracy == pytest.approx(0.5)


def test_threshold_one_predicts_everyone_negative():
    metrics = compute_metrics(y_true, probs, 1.0)
    assert metrics.confusion_matrix == [[3, 0], [3, 0]]
    assert metrics.recall == 0.0
    assert metrics.precision == 0.0  # undefined, reported as 0 rather than failing
    assert metrics.f1 == 0.0
    assert metrics.accuracy == pytest.approx(0.5)


def test_threshold_above_every_probability_is_still_scored():
    metrics = compute_metrics(y_true, probs, 7.0)
    assert metrics.confusion_matrix == [[3, 0], [3, 0]]
    assert metrics.threshold == 7.0


def test_a_probability_equal_to_the_threshold_is_positive():
    metrics = compute_metrics(np.array([0, 1]), np.array([0.3, 0.5]), 0.5)
    assert metrics.confusion_matrix == [[1, 0], [0, 1]]
    # ... and the one just below it is negative
    metrics = compute_metrics(np.array([0, 1]), np.array([0.3, 0.4999]), 0.5)
    assert metrics.confusion_matrix == [[1, 0], [1, 0]]


def test_ties_at_the_threshold_are_all_positive():
    tied = np.array([0.5, 0.5, 0.5, 0.5])
    metrics = compute_metrics(np.array([0, 1, 0, 1]), tied, 0.5)
    assert metrics.confusion_matrix == [[0, 2], [0, 2]]
    assert metrics.roc_auc == pytest.approx(0.5)


@pytest.mark.parametrize("method", ["f1", "youden"])
def test_constant_probabilities_give_a_usable_threshold(method):
    labels = np.array([0, 1, 0, 1, 1])
    threshold = choose_threshold(labels, np.full(5, 0.5), method)
    assert math.isfinite(threshold)
    assert 0.0 <= threshold <= 1.0


@pytest.mark.parametrize("method", ["f1", "youden"])
def test_perfectly_separable_scores_give_a_perfect_threshold(method):
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = choose_threshold(labels, scores, method)
    assert 0.2 < threshold <= 0.8
    assert compute_metrics(labels, scores, threshold).f1 == 1.0


@pytest.mark.parametrize("method", ["f1", "youden"])
def test_threshold_never_exceeds_one(method):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 50)
    labels[:2] = [0, 1]
    for scores in (rng.random(50), np.ones(50), np.round(rng.random(50), 1)):
        assert choose_threshold(labels, scores, method) <= 1.0


def test_tied_scores_choose_a_threshold_that_exists():
    labels = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.3, 0.3, 0.6, 0.6, 0.3, 0.6])
    for method in ("f1", "youden"):
        assert choose_threshold(labels, scores, method) in (0.3, 0.6, 1.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        choose_threshold(y_true, probs, "median")


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_validation_set(label):
    labels = np.full(5, label)
    scores = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    for method in ("f1", "youden"):
        try:
            threshold = choose_threshold(labels, scores, method)
        except ValueError:
            continue  # refusing is fine too
        assert not math.isnan(threshold)


@pytest.mark.parametrize("label", [0, 1])
def test_roc_auc_of_a_single_class_is_undefined(label):
    # ROC AUC needs both classes: scikit-learn returns NaN (older versions raised)
    try:
        metrics = compute_metrics(np.full(4, label), np.array([0.1, 0.4, 0.6, 0.9]), 0.5)
    except ValueError:
        return
    assert math.isnan(metrics.roc_auc)


def test_metrics_accept_float_labels_and_lists():
    labels = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    scores = [0.2, 0.6, 0.4, 0.9]
    metrics = compute_metrics(labels, np.array(scores), 0.5)
    assert metrics.confusion_matrix == [[1, 1], [1, 1]]
    assert metrics.scores().keys() == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        compute_metrics(y_true, probs[:-1], 0.5)
    with pytest.raises(ValueError):
        compute_metrics(y_true, np.array([0.1, 0.2, np.nan, 0.4, 0.7, 0.9]), 0.5)
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]), 0.5)
