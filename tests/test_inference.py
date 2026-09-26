"""Tests for inference routines."""
import numpy as np
import pytest
from psf import PSFModel, forward_backward, viterbi, particle_filter
from psf.states import IDX


@pytest.fixture
def small_model():
    np.random.seed(0)
    model = PSFModel(n_features=3)
    model.mu = np.random.randn(6, 3)
    model.log_sigma = np.log(np.full((6, 3), 0.5))
    return model


def test_forward_backward_shapes(small_model):
    T = 20
    y = np.random.randn(T, 3)
    out = forward_backward(small_model, y)
    assert out["gamma"].shape == (T, 6)
    assert out["log_alpha"].shape == (T, 6)
    assert out["log_beta"].shape == (T, 6)
    assert out["xi"].shape == (T - 1, 6, 6)


def test_posterior_sums_to_one(small_model):
    T = 15
    y = np.random.randn(T, 3)
    out = forward_backward(small_model, y)
    sums = out["gamma"].sum(axis=1)
    assert np.allclose(sums, 1.0)


def test_log_likelihood_finite(small_model):
    y = np.random.randn(10, 3)
    out = forward_backward(small_model, y)
    assert np.isfinite(out["log_lik"])


def test_viterbi_returns_valid_path(small_model):
    T = 20
    y = np.random.randn(T, 3)
    path = viterbi(small_model, y)
    assert path.shape == (T,)
    assert np.all((path >= 0) & (path < 6))


def test_particle_filter_shape(small_model):
    T = 15
    y = np.random.randn(T, 3)
    out = particle_filter(small_model, y, n_particles=200)
    assert out["filtered_probs"].shape == (T, 6)
    assert out["ess"].shape == (T,)


def test_particle_filter_probs_sum_to_one(small_model):
    y = np.random.randn(10, 3)
    out = particle_filter(small_model, y, n_particles=500)
    sums = out["filtered_probs"].sum(axis=1)
    # Filtering distribution should sum to ~1 each step
    assert np.allclose(sums, 1.0, atol=0.05)


def test_return_probabilities(small_model):
    T = 10
    y = np.random.randn(T, 3)
    out = forward_backward(small_model, y)
    p = __import__("psf").return_probabilities(small_model, out["gamma"], horizon=5)
    assert p.shape == (T, 5)
    assert np.all((p >= 0) & (p <= 1))