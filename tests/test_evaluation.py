"""Tests for evaluation metrics."""
import numpy as np
from psf import evaluate_predictions, concordance_index


def test_evaluate_perfect_classifier():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = y.astype(float)
    m = evaluate_predictions(y, p)
    assert np.isclose(m["accuracy"], 1.0)
    assert np.isclose(m["auc"], 1.0)
    assert np.isclose(m["f1"], 1.0)


def test_evaluate_random_classifier():
    np.random.seed(0)
    y = np.random.randint(0, 2, size=200)
    p = np.random.rand(200)
    m = evaluate_predictions(y, p)
    assert 0.4 < m["auc"] < 0.6


def test_concordance_index():
    times = np.array([1, 2, 3, 4, 5])
    events = np.array([1, 1, 1, 1, 1])
    risks = np.array([5, 4, 3, 2, 1])  # higher risk = shorter time
    c = concordance_index(times, events, risks)
    assert np.isclose(c, 1.0)