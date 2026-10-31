import numpy as np
import pytest

from diabetes_classifier.evaluate import choose_threshold, compute_metrics

y_true = np.array([0, 0, 0, 1, 1, 1])
probs = np.array([0.1, 0.2, 0.45, 0.4, 0.7, 0.9])


def test_fixed_threshold_is_returned_as_is():
    assert choose_threshold(y_true, probs, 0.5) == 0.5


@pytest.mark.parametrize("method", ["f1", "youden"])
def test_tuned_threshold_is_one_of_the_probabilities(method):
    threshold = choose_threshold(y_true, probs, method)
    assert threshold in probs
    assert compute_metrics(y_true, probs, threshold).f1 >= compute_metrics(y_true, probs, 0.5).f1


def test_metrics_at_threshold():
    metrics = compute_metrics(y_true, probs, 0.5)
    assert metrics.confusion_matrix == [[3, 0], [1, 2]]
    assert metrics.accuracy == pytest.approx(5 / 6)
    assert metrics.precision == 1.0
    assert metrics.recall == pytest.approx(2 / 3)
