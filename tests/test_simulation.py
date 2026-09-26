"""Tests for synthetic data generation."""
import numpy as np
from psf import PSFModel, simulate_cohort


def test_simulation_shapes():
    model = PSFModel(n_features=3)
    model.mu = np.random.randn(6, 3)
    model.log_sigma = np.zeros((6, 3))
    data = simulate_cohort(model, n_students=10, max_time=20)
    assert len(data["observations"]) == 10
    assert len(data["states"]) == 10
    assert data["final_label"].shape == (10,)


def test_simulation_reproducible():
    model = PSFModel(n_features=2)
    model.mu = np.random.randn(6, 2)
    model.log_sigma = np.zeros((6, 2))
    d1 = simulate_cohort(model, n_students=5, max_time=10, rng=np.random.default_rng(42))
    d2 = simulate_cohort(model, n_students=5, max_time=10, rng=np.random.default_rng(42))
    for a, b in zip(d1["observations"], d2["observations"]):
        assert np.allclose(a, b)