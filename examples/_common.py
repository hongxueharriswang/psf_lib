"""Shared helpers for examples."""
import numpy as np
from psf import PSFModel
from psf.states import IDX


def build_true_model(n_features: int = 6) -> PSFModel:
    """Build a synthetic 'true' model for testing."""
    m = PSFModel(n_features=n_features)
    A_logits = np.full((6, 6), -np.inf)
    A_logits[IDX.S1, IDX.S1] = 1.0
    A_logits[IDX.S1, IDX.S2] = 3.0
    A_logits[IDX.S2, IDX.S2] = 3.0
    A_logits[IDX.S2, IDX.S3_NO_INTENT] = 0.0
    A_logits[IDX.S2, IDX.S3_INTENT] = 1.0
    A_logits[IDX.S2, IDX.S4] = -2.0
    A_logits[IDX.S2, IDX.S5] = -0.5
    A_logits[IDX.S3_NO_INTENT, IDX.S3_NO_INTENT] = 2.0
    A_logits[IDX.S3_NO_INTENT, IDX.S3_INTENT] = 0.0
    A_logits[IDX.S3_NO_INTENT, IDX.S4] = 0.0
    A_logits[IDX.S3_INTENT, IDX.S3_INTENT] = 1.0
    A_logits[IDX.S3_INTENT, IDX.S2] = 0.5
    A_logits[IDX.S3_INTENT, IDX.S4] = -1.0
    A_logits[IDX.S4, IDX.S4] = 0.0
    A_logits[IDX.S5, IDX.S5] = 0.0
    m.A_logits = A_logits
    m._refresh_A()

    rng = np.random.default_rng(42)
    m.mu = rng.normal(loc=0, scale=1, size=(6, n_features))
    m.mu[IDX.S2] += np.array([1.0, 0.8, 0.0, 0.0, 0.0, -0.5])
    m.mu[IDX.S3_NO_INTENT] += np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 1.0])
    m.mu[IDX.S3_INTENT] += np.array([-0.5, -0.3, 0.5, 0.5, 0.8, 0.8])
    m.mu[IDX.S4] += np.array([-2.0, -1.0, -0.5, -0.5, -0.5, 1.5])
    m.mu[IDX.S5] += np.array([0.5, 1.2, 0.2, 0.0, 0.0, -1.0])
    m.log_sigma = np.log(np.full((6, n_features), 0.7))
    return m