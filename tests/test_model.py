"""Tests for PSFModel."""
import numpy as np
import pytest
from psf import PSFModel
from psf.states import ALLOWED_TRANSITIONS, IDX


def test_initialization():
    model = PSFModel(n_features=6)
    assert model.A.shape == (6, 6)
    assert model.pi.shape == (6,)
    assert model.mu.shape == (6, 6)
    assert np.isclose(model.pi[IDX.S1], 1.0)


def test_transition_matrix_valid():
    model = PSFModel(n_features=3)
    row_sums = model.A.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_absorbing_states():
    model = PSFModel(n_features=3)
    assert np.isclose(model.A[IDX.S4, IDX.S4], 1.0)
    assert np.isclose(model.A[IDX.S5, IDX.S5], 1.0)
    assert np.isclose(model.A[IDX.S4].sum(), 1.0)
    assert np.isclose(model.A[IDX.S5].sum(), 1.0)


def test_structural_zeros():
    model = PSFModel(n_features=3)
    mask = ALLOWED_TRANSITIONS == 0
    assert np.allclose(model.A[mask], 0.0)


def test_log_emission_shape():
    model = PSFModel(n_features=4)
    y = np.random.randn(10, 4)
    log_b = model.log_emission_prob(y)
    assert log_b.shape == (10, 6)


def test_log_emission_with_mask():
    model = PSFModel(n_features=4)
    y = np.random.randn(10, 4)
    mask = np.ones_like(y, dtype=bool)
    mask[0, 0] = False
    log_b = model.log_emission_prob(y, mask)
    assert log_b.shape == (10, 6)


def test_n_params_positive():
    model = PSFModel(n_features=6)
    assert model.n_params() > 0