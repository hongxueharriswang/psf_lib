"""The PSFModel: joint-state HMM with latent Intent-to-Return.

Parameters
----------
A : (K, K) transition matrix, K = 6 joint states
pi : (K,) initial state distribution
mu : (K, D) emission means
log_sigma : (K, D) emission log standard deviations

Structural constraints
----------------------
- Transitions follow ALLOWED_TRANSITIONS (see states.py)
- S4 (Withdrawn) and S5 (Completed) are absorbing
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .states import (
    ALLOWED_TRANSITIONS,
    IDX,
    N_JOINT_STATES,
)
from .utils import softmax_rows


@dataclass
class PSFModel:
    n_states: int = N_JOINT_STATES
    n_features: int = 1

    # Parameters (initialized in __post_init__)
    A: np.ndarray = field(default=None)          # (K, K)
    pi: np.ndarray = field(default=None)         # (K,)
    mu: np.ndarray = field(default=None)         # (K, D)
    log_sigma: np.ndarray = field(default=None)  # (K, D)

    # Structure
    allowed: np.ndarray = field(default_factory=lambda: ALLOWED_TRANSITIONS.copy())

    # Free transition logits (used only during EM)
    A_logits: np.ndarray = field(default=None)

    def __post_init__(self):
        K = self.n_states
        D = self.n_features
        if self.A_logits is None:
            # Initialize with weak preference for staying in current state
            self.A_logits = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    if self.allowed[i, j] > 0:
                        self.A_logits[i, j] = 1.0 if i == j else 0.0
            self._refresh_A()
        if self.pi is None:
            self.pi = np.zeros(K)
            self.pi[IDX.S1] = 1.0
        if self.mu is None:
            self.mu = np.zeros((K, D))
        if self.log_sigma is None:
            self.log_sigma = np.zeros((K, D))

    # ---------- Parameter refresh ----------

    def _refresh_A(self):
        """Recompute A from A_logits respecting structural zeros."""
        masked = np.where(self.allowed > 0, self.A_logits, -np.inf)
        self.A = softmax_rows(masked)
        # Ensure absorbing rows are exactly absorbing
        self.A[IDX.S4] = 0.0
        self.A[IDX.S4, IDX.S4] = 1.0
        self.A[IDX.S5] = 0.0
        self.A[IDX.S5, IDX.S5] = 1.0
        # Enforce structural zeros (for safety)
        self.A = self.A * self.allowed
        row_sums = self.A.sum(axis=1, keepdims=True)
        self.A = self.A / np.maximum(row_sums, 1e-12)

    def set_transition_logits(self, A_logits: np.ndarray):
        self.A_logits = A_logits
        self._refresh_A()

    # ---------- Emission likelihood ----------

    def log_emission_prob(
        self,
        observations: np.ndarray,
        mask: np.ndarray | None = None,
        ) -> np.ndarray:
        """Log p(y_t | z_t = k) for all t, k.

        Parameters
        ----------
        observations : (T, D)
        mask : (T, D) boolean, True = observed, False = missing.
               If None, all features are treated as observed.

        Returns
        -------
        log_b : (T, K)

        Raises
        ------
        ValueError
            If `observations` or `mask` has an incompatible shape.
        """
        if observations.ndim != 2:
            raise ValueError(
                f"observations must be 2D (T, D); got shape {observations.shape}"
            )
        _, D = observations.shape
        if D != self.n_features:
            raise ValueError(
                f"observations has D={D} features, but model was "
                f"initialized with n_features={self.n_features}"
            )

        if mask is None:
            mask = np.ones_like(observations, dtype=bool)
        elif mask.shape != observations.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match "
                f"observations shape {observations.shape}"
            )
        mask = mask.astype(float)

        # (T, K, D) broadcast; K = self.n_states
        diff = observations[:, None, :] - self.mu[None, :, :]
        var = np.exp(2.0 * self.log_sigma)[None, :, :]
        log_norm = -0.5 * np.log(2.0 * np.pi) - self.log_sigma[None, :, :]
        quad = -0.5 * (diff ** 2) / var
        per_feature = (log_norm + quad) * mask[:, None, :]
        log_b = per_feature.sum(axis=2)  # (T, K)
        return log_b
    # ---------- Convenience ----------

    def n_params(self) -> int:
        n_A = int(self.allowed.sum())
        n_pi = self.n_states - 1
        n_emit = 2 * self.n_states * self.n_features
        return n_A + n_pi + n_emit

    def __repr__(self):
        return (
            f"PSFModel(K={self.n_states}, D={self.n_features}, "
            f"n_params={self.n_params()})"
        )