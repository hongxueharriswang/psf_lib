"""State definitions and structural constraints for the PSF."""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class PersistenceState(IntEnum):
    """Five observed persistence states."""
    REGISTERED = 0
    PROGRESSING = 1
    STOPPED_OUT = 2
    WITHDRAWN = 3
    COMPLETED = 4


class JointState(IntEnum):
    """Joint (persistence, intent) latent state used by the HMM.

    Intent is only defined when the persistence state is Stopped-Out.
    For other states, intent is collapsed to a single value.
    """
    S1_REGISTERED = 0
    S2_PROGRESSING = 1
    S3_STOPPED_OUT_NO_INTENT = 2
    S3_STOPPED_OUT_INTENT = 3
    S4_WITHDRAWN = 4
    S5_COMPLETED = 5


N_JOINT_STATES = 6

JOINT_STATE_LABELS = (
    "S1: Registered",
    "S2: Progressing",
    "S3: Stopped-Out (I=0)",
    "S3: Stopped-Out (I=1)",
    "S4: Withdrawn",
    "S5: Completed",
)

# Structural mask: ALLOWED[i, j] == 1 iff transition i -> j is permitted.
# Rows/cols indexed by JointState.
ALLOWED_TRANSITIONS: np.ndarray = np.array(
    [
        #  S1  S2  S3(0) S3(1) S4  S5
        [1,  1,   0,    0,    0,  0],   # from S1
        [0,  1,   1,    1,    1,  1],   # from S2
        [0,  0,   1,    1,    1,  0],   # from S3(I=0)
        [0,  1,   0,    1,    1,  0],   # from S3(I=1)
        [0,  0,   0,    0,    1,  0],   # from S4 (absorbing)
        [0,  0,   0,    0,    0,  1],   # from S5 (absorbing)
    ],
    dtype=float,
)

# Mapping from JointState to PersistenceState (for collapsing posteriors)
JOINT_TO_PERSISTENCE = np.array(
    [
        PersistenceState.REGISTERED,
        PersistenceState.PROGRESSING,
        PersistenceState.STOPPED_OUT,
        PersistenceState.STOPPED_OUT,
        PersistenceState.WITHDRAWN,
        PersistenceState.COMPLETED,
    ]
)


@dataclass(frozen=True)
class TransitionIndices:
    """Named indices into the joint transition matrix."""
    S1: int = JointState.S1_REGISTERED
    S2: int = JointState.S2_PROGRESSING
    S3_NO_INTENT: int = JointState.S3_STOPPED_OUT_NO_INTENT
    S3_INTENT: int = JointState.S3_STOPPED_OUT_INTENT
    S4: int = JointState.S4_WITHDRAWN
    S5: int = JointState.S5_COMPLETED


IDX = TransitionIndices()


def collapse_posterior(gamma: np.ndarray) -> np.ndarray:
    """Collapse joint-state posterior (T, 6) to persistence posterior (T, 5).

    The two S3 intent variants are summed into a single STOPPED_OUT column.
    """
    T = gamma.shape[0]
    out = np.zeros((T, 5))
    out[:, PersistenceState.REGISTERED] = gamma[:, JointState.S1_REGISTERED]
    out[:, PersistenceState.PROGRESSING] = gamma[:, JointState.S2_PROGRESSING]
    out[:, PersistenceState.STOPPED_OUT] = (
        gamma[:, JointState.S3_STOPPED_OUT_NO_INTENT]
        + gamma[:, JointState.S3_STOPPED_OUT_INTENT]
    )
    out[:, PersistenceState.WITHDRAWN] = gamma[:, JointState.S4_WITHDRAWN]
    out[:, PersistenceState.COMPLETED] = gamma[:, JointState.S5_COMPLETED]
    return out


def intent_posterior(gamma: np.ndarray) -> np.ndarray:
    """Return P(I_t = 1 | y_{1:T}) at each time step.

    Defined as the posterior probability of being in S3 with intent,
    normalized by the total S3 posterior.
    """
    p_s3 = (
        gamma[:, JointState.S3_STOPPED_OUT_NO_INTENT]
        + gamma[:, JointState.S3_STOPPED_OUT_INTENT]
    )
    p_intent = gamma[:, JointState.S3_STOPPED_OUT_INTENT]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(p_s3 > 1e-12, p_intent / p_s3, np.nan)
    return out