"""Tests for competing-risk survival models."""
import numpy as np
from psf import CompetingRiskSurvival


def test_fit_and_predict():
    np.random.seed(0)
    N = 100
    X = np.random.randn(N, 3)
    t = np.random.randint(1, 25, size=N)
    event = np.random.choice([0, 1, 2, 3], size=N, p=[0.2, 0.3, 0.3, 0.2])

    model = CompetingRiskSurvival()
    model.fit(X, t, event)
    cif = model.predict_cumulative_incidence(X, horizon=12)
    assert cif.shape == (N, 12, 3)
    assert np.all((cif >= 0) & (cif <= 1))