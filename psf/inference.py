"""Inference: forward-backward, Viterbi, particle filter, return probabilities."""
from __future__ import annotations

import numpy as np

from .model import PSFModel
from .states import IDX
from .utils import logsumexp


def forward_backward(
    model: PSFModel,
    observations: np.ndarray,
    mask: np.ndarray | None = None,
    return_xi: bool = True,
) -> dict:
    """Log-space forward-backward.

    Returns dict with keys:
        log_alpha : (T, K)
        log_beta  : (T, K)
        gamma     : (T, K) posterior P(z_t | y_{1:T})
        xi        : (T-1, K, K) if return_xi else None
        log_lik   : scalar log p(y_{1:T})
    """
    T = observations.shape[0]
    K = model.n_states
    log_A = np.log(np.maximum(model.A, 1e-300))
    log_pi = np.log(np.maximum(model.pi, 1e-300))
    log_b = model.log_emission_prob(observations, mask)

    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, T):
        log_alpha[t] = log_b[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_A, axis=0
        )

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_A + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1
        )

    log_lik = logsumexp(log_alpha[-1], axis=0)

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi = None
    if return_xi and T > 1:
        log_xi = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + (log_b[1:] + log_beta[1:])[:, None, :]
            - log_lik
        )
        xi = np.exp(log_xi)

    return {
        "log_alpha": log_alpha,
        "log_beta": log_beta,
        "gamma": gamma,
        "xi": xi,
        "log_lik": float(log_lik),
    }


def viterbi(model: PSFModel, observations: np.ndarray, mask=None) -> np.ndarray:
    """Most likely joint-state sequence."""
    T = observations.shape[0]
    K = model.n_states
    log_A = np.log(np.maximum(model.A, 1e-300))
    log_pi = np.log(np.maximum(model.pi, 1e-300))
    log_b = model.log_emission_prob(observations, mask)

    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + log_b[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = log_b[t] + np.max(scores, axis=0)
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def particle_filter(
    model: PSFModel,
    observations: np.ndarray,
    mask: np.ndarray | None = None,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap particle filter for online state estimation.

    Returns
    -------
    filtered_probs : (T, K) filtering distribution P(z_t | y_{1:t})
    ess : (T,) effective sample sizes
    """
    if rng is None:
        rng = np.random.default_rng(0)
    T = observations.shape[0]
    K = model.n_states
    N = n_particles

    particles = rng.choice(K, size=N, p=model.pi)
    log_w = np.full(N, -np.log(N))
    filtered = np.zeros((T, K))
    ess = np.zeros(T)

    log_b_all = model.log_emission_prob(observations, mask)

    for t in range(T):
        # Weight update
        log_w = log_w + log_b_all[t][particles]
        log_w -= logsumexp(log_w)
        w = np.exp(log_w)

        # Store filtering distribution
        for k in range(K):
            filtered[t, k] = w[particles == k].sum()
        ess[t] = 1.0 / np.sum(w ** 2)

        # Resample if needed
        if ess[t] < ess_threshold * N:
            idx = rng.choice(N, size=N, p=w)
            particles = particles[idx]
            log_w = np.full(N, -np.log(N))

        # Propagate (except after last observation)
        if t < T - 1:
            A_rows = model.A[particles]
            # Sample next state per particle
            new_particles = np.array(
                [rng.choice(K, p=A_rows[i]) for i in range(N)]
            )
            particles = new_particles

    return {"filtered_probs": filtered, "ess": ess}


def return_probabilities(
    model: PSFModel,
    gamma: np.ndarray,
    horizon: int,
    progress_state: int = IDX.S2,
) -> np.ndarray:
    """P(Z_{t+h} = Progressing | y_{1:t}) for h = 1..horizon.

    Uses the HMM transition matrix and the filtering posterior.
    Returns (T, horizon) array.
    """
    T = gamma.shape[0]
    K = model.n_states
    A_h = np.eye(K)
    out = np.zeros((T, horizon))
    for h in range(1, horizon + 1):
        A_h = A_h @ model.A
        out[:, h - 1] = gamma @ A_h[:, progress_state]
    return out


def withdrawal_probabilities(
    model: PSFModel,
    gamma: np.ndarray,
    horizon: int,
    withdrawn_state: int = IDX.S4,
) -> np.ndarray:
    """P(Z_{t+h} = Withdrawn | y_{1:t}) for h = 1..horizon."""
    T = gamma.shape[0]
    K = model.n_states
    A_h = np.eye(K)
    out = np.zeros((T, horizon))
    for h in range(1, horizon + 1):
        A_h = A_h @ model.A
        out[:, h - 1] = gamma @ A_h[:, withdrawn_state]
    return out