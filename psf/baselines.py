"""Baseline models for comparison against the PSF."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InactivityThresholdClassifier:
    """Predicts withdrawal if no activity for `threshold` time units."""
    threshold: int = 2

    def fit(self, X, y, inactivity_features: np.ndarray):
        """`inactivity_features` is the time since last activity at label time."""
        self._inactivity = inactivity_features
        return self

    def predict_proba(self, inactivity_features: np.ndarray) -> np.ndarray:
        p = (inactivity_features >= self.threshold).astype(float)
        return np.column_stack([1 - p, p])


@dataclass
class LogisticDropoutModel:
    """Static binary logistic regression baseline."""
    C: float = 1.0

    def fit(self, X, y_binary):
        from sklearn.linear_model import LogisticRegression
        self.clf_ = LogisticRegression(C=self.C, max_iter=2000)
        self.clf_.fit(X, y_binary)
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(X)


@dataclass
class SurvivalWithoutIntent:
    """Discrete-time survival model without the Intent-to-Return variable.

    Wraps CompetingRiskSurvival but only uses covariates that are
    available under the no-intent assumption.
    """
    max_time: int = 24

    def fit(self, X, t, event):
        from .survival import CompetingRiskSurvival
        self.model_ = CompetingRiskSurvival().fit(X, t, event)
        return self

    def predict_cumulative_incidence(self, X, horizon=None):
        return self.model_.predict_cumulative_incidence(X, horizon=horizon)