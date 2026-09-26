"""Tests for state definitions and structural constraints."""
import numpy as np
from psf.states import (
    PersistenceState,
    JointState,
    ALLOWED_TRANSITIONS,
    JOINT_TO_PERSISTENCE,
    collapse_posterior,
    intent_posterior,
)


def test_enum_values():
    assert PersistenceState.REGISTERED == 0
    assert PersistenceState.PROGRESSING == 1
    assert PersistenceState.STOPPED_OUT == 2
    assert PersistenceState.WITHDRAWN == 3
    assert PersistenceState.COMPLETED == 4


def test_allowed_transitions_shape():
    assert ALLOWED_TRANSITIONS.shape == (6, 6)
    # Absorbing states
    assert ALLOWED_TRANSITIONS[JointState.S4_WITHDRAWN, JointState.S4_WITHDRAWN] == 1
    assert ALLOWED_TRANSITIONS[JointState.S5_COMPLETED, JointState.S5_COMPLETED] == 1
    # No transition out of absorbing states
    assert ALLOWED_TRANSITIONS[JointState.S4_WITHDRAWN].sum() == 1
    assert ALLOWED_TRANSITIONS[JointState.S5_COMPLETED].sum() == 1


def test_collapse_posterior():
    gamma = np.zeros((3, 6))
    gamma[:, JointState.S1_REGISTERED] = 1.0
    gamma[:, JointState.S2_PROGRESSING] = 2.0
    gamma[:, JointState.S3_STOPPED_OUT_NO_INTENT] = 3.0
    gamma[:, JointState.S3_STOPPED_OUT_INTENT] = 4.0
    gamma[:, JointState.S4_WITHDRAWN] = 5.0
    gamma[:, JointState.S5_COMPLETED] = 6.0
    out = collapse_posterior(gamma)
    assert out.shape == (3, 5)
    assert np.allclose(out[:, PersistenceState.STOPPED_OUT], 7.0)


def test_intent_posterior():
    gamma = np.zeros((2, 6))
    gamma[0, JointState.S3_STOPPED_OUT_NO_INTENT] = 0.3
    gamma[0, JointState.S3_STOPPED_OUT_INTENT] = 0.7
    gamma[1, JointState.S2_PROGRESSING] = 1.0
    out = intent_posterior(gamma)
    assert np.isclose(out[0], 0.7)
    assert np.isnan(out[1])