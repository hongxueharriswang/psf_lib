"""Discrete-time competing-risk survival models.

Event codes:
    0 = censored / still at risk
    1 = return to Progressing
    2 = withdrawal
    3 = completion
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CompetingRiskSurvival:
    # Convention: 0 = censored, 1 = completed, 2 = withdrawn, 3 = stopped-out
    causes: tuple = (1, 2, 3)
    max_time: int | None = None
    models: dict = field(default_factory=dict)   # cause -> fitted LogisticRegression
    time_grid_: np.ndarray | None = None
    include_time: bool = True

    def _expand(self, X, t, event):
        """Expand (X, t, event) into person-period format."""
        N = len(t)
        rows_X, rows_t, rows_y = [], [], []
        for i in range(N):
            for s in range(1, t[i] + 1):
                y = event[i] if s == t[i] else 0
                rows_X.append(X[i])
                rows_t.append(s)
                rows_y.append(y)
        return (
            np.asarray(rows_X, dtype=float),
            np.asarray(rows_t, dtype=float),
            np.asarray(rows_y, dtype=int),
        )

    def fit(self, X, t, event):
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=int)
        event = np.asarray(event, dtype=int)
        self.max_time = int(t.max()) if self.max_time is None else self.max_time

        Xp, tp, yp = self._expand(X, t, event)
        if self.include_time:
            Xp_design = np.column_stack([Xp, tp / self.max_time])
        else:
            Xp_design = Xp

        for k in self.causes:
            y_bin = (yp == k).astype(int)
            if y_bin.sum() < 5:
                # Not enough events; store a constant model
                self.models[k] = ("constant", float(y_bin.mean()))
                continue
            clf = LogisticRegression(max_iter=2000, solver="lbfgs")
            clf.fit(Xp_design, y_bin)
            self.models[k] = clf
        self.time_grid_ = np.arange(1, self.max_time + 1)
        return self

    def predict_hazard(self, X, times=None):
        """Return hazard array (N, T, K) over times 1..T."""
        X = np.asarray(X, dtype=float)
        if times is None:
            times = self.time_grid_
        N = len(X)
        T = len(times)
        K = len(self.causes)
        H = np.zeros((N, T, K))
        for ki, k in enumerate(self.causes):
            m = self.models[k]
            for ti, s in enumerate(times):
                if self.include_time:
                    Xd = np.column_stack([X, np.full(N, s / self.max_time)])
                else:
                    Xd = X
                if isinstance(m, tuple):  # constant
                    H[:, ti, ki] = m[1]
                else:
                    H[:, ti, ki] = m.predict_proba(Xd)[:, 1]
        return H

    def predict_cumulative_incidence(self, X, horizon=None):
        """Cumulative incidence function for each cause.

        Returns (N, T, K) where T = horizon (or max_time).
        """
        X = np.asarray(X, dtype=float)
        if horizon is None:
            horizon = self.max_time
        times = np.arange(1, horizon + 1)
        H = self.predict_hazard(X, times)         # (N, T, K)
        # Cause-specific survival: S_k(t) = prod_{s<=t} (1 - h_k(s))
        S_k = np.cumprod(1.0 - H, axis=1)         # (N, T, K)
        # Overall survival: S(t) = prod_k S_k(t)
        S_overall = np.prod(S_k, axis=2)          # (N, T)
        # CIF for cause k: sum_{s<=t} S_overall(s-1) * h_k(s)
        S_prev = np.concatenate([np.ones((X.shape[0], 1)), S_overall[:, :-1]], axis=1)
        CIF = np.cumsum(S_prev[:, :, None] * H, axis=1)
        return CIF