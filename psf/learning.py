"""Baum-Welch (EM) parameter estimation for the PSF."""
from __future__ import annotations

import numpy as np

from .inference import forward_backward
from .model import PSFModel
from .states import ALLOWED_TRANSITIONS, IDX
from .utils import ensure_positive


def baum_welch(
    model: PSFModel,
    sequences: List[np.ndarray],
    masks: Optional[List[Optional[np.ndarray]]] = None,
    n_iter: int = 50,
    tol: float = 1e-4,
    verbose: bool = False,
    sigma_floor: float = 1e-3,
) -> PSFModel:
    """Fit PSFModel parameters via EM (Baum-Welch).

    The per-iteration total log-likelihood is recorded on
    ``model.em_history_`` as a list of floats, which can be plotted with
    :func:`psf.viz.plot_em_convergence`.
    """
    if masks is None:
        masks = [None] * len(sequences)

    K = model.n_states
    D = model.n_features
    prev_ll = -np.inf
    history: List[float] = []

    for it in range(n_iter):
        # ---------------- E-step ----------------
        cache = []
        total_ll = 0.0

        for seq, m in zip(sequences, masks):
            if m is None:
                m = np.ones_like(seq, dtype=bool)
            m_f = m.astype(float)

            out = forward_backward(model, seq, m, return_xi=True)
            cache.append((seq, m_f, out["gamma"], out["xi"]))
            total_ll += out["log_lik"]

        history.append(total_ll)          # <-- record history

        # ---------------- Accumulators ----------------
        pi_acc = np.zeros(K)
        A_acc = np.zeros((K, K))
        mu_num = np.zeros((K, D))
        mu_den = np.zeros((K, D))

        for seq, m_f, gamma, xi in cache:
            pi_acc += gamma[0]
            if xi is not None:
                A_acc += xi.sum(axis=0)
            mu_num += gamma.T @ (seq * m_f)
            mu_den += gamma.T @ m_f

        # ---------------- M-step: initial distribution ----------------
        pi_new = ensure_positive(pi_acc)
        pi_new /= pi_new.sum()
        model.pi = pi_new

        # ---------------- M-step: transition matrix ----------------
        A_new = np.where(ALLOWED_TRANSITIONS > 0, A_acc, 0.0)
        row_sums = A_new.sum(axis=1, keepdims=True)
        A_new = A_new / np.maximum(row_sums, 1e-12)
        A_new[IDX.S4] = 0.0
        A_new[IDX.S4, IDX.S4] = 1.0
        A_new[IDX.S5] = 0.0
        A_new[IDX.S5, IDX.S5] = 1.0

        with np.errstate(divide="ignore"):
            A_logits_new = np.where(
                ALLOWED_TRANSITIONS > 0,
                np.log(np.maximum(A_new, 1e-12)),
                -np.inf,
            )
        model.A_logits = A_logits_new
        model._refresh_A()

        # ---------------- M-step: emissions ----------------
        mu_new = mu_num / np.maximum(mu_den, 1e-12)
        model.mu = mu_new

        var_num = np.zeros((K, D))
        var_den = np.zeros((K, D))
        for seq, m_f, gamma, _ in cache:
            resid_sq = (seq[:, None, :] - mu_new[None, :, :]) ** 2
            contrib = gamma[:, :, None] * resid_sq * m_f[:, None, :]
            var_num += contrib.sum(axis=0)
            var_den += (gamma[:, :, None] * m_f[:, None, :]).sum(axis=0)

        var_new = var_num / np.maximum(var_den, 1e-12)
        var_new = np.maximum(var_new, sigma_floor ** 2)
        model.log_sigma = np.log(np.sqrt(var_new))

        # ---------------- Convergence check ----------------
        if verbose:
            print(f"[EM] iter {it + 1:3d}  log-lik = {total_ll:.4f}")
        if abs(total_ll - prev_ll) < tol:
            if verbose:
                print(f"[EM] converged after {it + 1} iterations.")
            break
        prev_ll = total_ll

    model.em_history_ = history         # <-- attach to model
    return model