"""Synthetic data generation for validating the PSF."""
from __future__ import annotations

import numpy as np

from .model import PSFModel
from .states import IDX, JOINT_TO_PERSISTENCE


def simulate_cohort(
    model: PSFModel,
    n_students: int = 500,
    max_time: int = 36,
    missing_rate: float = 0.2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate a cohort of students under the PSF.

    Returns
    -------
    dict with keys:
        observations : list of (T_i, D) arrays
        masks        : list of (T_i, D) boolean arrays
        states       : list of (T_i,) joint-state trajectories
        persistence  : list of (T_i,) persistence-state trajectories
        final_label  : (N,) operational label (0=Completed, 1=Withdrawn,
                       2=Stopped-Out, 3=Censored)
        durations    : (N,) end times
    """
    if rng is None:
        rng = np.random.default_rng(0)
    K = model.n_states
    D = model.n_features

    observations, masks, states, persistence = [], [], [], []
    final_label = np.full(n_students, 3, dtype=int)
    durations = np.zeros(n_students, dtype=int)

    for n in range(n_students):
        # Sample initial state
        z = rng.choice(K, p=model.pi)
        obs_seq, mask_seq, state_seq = [], [], []
        for t in range(max_time):
            state_seq.append(z)
            # Sample observation
            mu = model.mu[z]
            sigma = np.exp(model.log_sigma[z])
            y = mu + sigma * rng.standard_normal(D)
            mask = rng.random(D) > missing_rate
            obs_seq.append(y)
            mask_seq.append(mask)
            # Transition
            if z in (IDX.S4, IDX.S5):
                # Absorbing: stop simulation
                break
            z = rng.choice(K, p=model.A[z])
        observations.append(np.asarray(obs_seq))
        masks.append(np.asarray(mask_seq, dtype=bool))
        states.append(np.asarray(state_seq))
        persistence.append(JOINT_TO_PERSISTENCE[np.asarray(state_seq)])
        durations[n] = len(state_seq)
        # Final label
        last = state_seq[-1]
        if last == IDX.S5:
            final_label[n] = 0  # Completed
        elif last == IDX.S4:
            final_label[n] = 1  # Withdrawn
        elif last in (IDX.S3_NO_INTENT, IDX.S3_INTENT):
            if last == IDX.S3_INTENT:
                final_label[n] = 2  # Stopped-Out with intent
            else:
                final_label[n] = 1  # No intent -> operational withdrawal
        else:
            final_label[n] = 3  # Censored / still enrolled
    return {
        "observations": observations,
        "masks": masks,
        "states": states,
        "persistence": persistence,
        "final_label": final_label,
        "durations": durations,
    }