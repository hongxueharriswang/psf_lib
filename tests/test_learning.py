"""Tests for Baum-Welch."""
import numpy as np
import pytest
from psf import PSFModel, baum_welch, forward_backward


def test_em_improves_log_likelihood():
    np.random.seed(0)
    model = PSFModel(n_features=2)
    model.mu = np.array([[0, 0], [1, 1], [-1, -1], [0, 1], [1, 0], [0.5, 0.5]])
    model.log_sigma = np.log(np.full((6, 2), 0.8))

    obs = [np.random.randn(20, 2) for _ in range(5)]
    ll_before = sum(forward_backward(model, o)["log_lik"] for o in obs)

    baum_welch(model, obs, n_iter=5, verbose=False)
    ll_after = sum(forward_backward(model, o)["log_lik"] for o in obs)

    assert ll_after > ll_before


def test_em_respects_structural_zeros():
    np.random.seed(0)
    model = PSFModel(n_features=2)
    model.mu = np.random.randn(6, 2)
    model.log_sigma = np.zeros((6, 2))
    obs = [np.random.randn(15, 2) for _ in range(3)]
    baum_welch(model, obs, n_iter=3)
    from psf.states import ALLOWED_TRANSITIONS
    mask = ALLOWED_TRANSITIONS == 0
    assert np.allclose(model.A[mask], 0.0)


def test_em_preserves_absorbing_states():
    np.random.seed(0)
    model = PSFModel(n_features=2)
    model.mu = np.random.randn(6, 2)
    model.log_sigma = np.zeros((6, 2))
    obs = [np.random.randn(15, 2) for _ in range(3)]
    baum_welch(model, obs, n_iter=3)
    from psf.states import IDX
    assert np.isclose(model.A[IDX.S4, IDX.S4], 1.0)
    assert np.isclose(model.A[IDX.S5, IDX.S5], 1.0)