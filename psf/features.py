"""Feature extraction from student records."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureExtractor:
    """Extract per-timestep feature vectors for the PSF HMM.

    Features (default):
        0: log1p(activity_count)
        1: submission_occurred (0/1)
        2: extension_requested (0/1)
        3: tutor_contact (0/1)
        4: future_registration (0/1)
        5: log1p(time_since_last_activity)
    """
    log1p_counts: bool = True

    def transform_timestep(
        self,
        activity_count: float,
        submission: float,
        extension: float,
        tutor_contact: float,
        future_registration: float,
        time_since_last_activity: float,
    ) -> np.ndarray:
        a = np.log1p(activity_count) if self.log1p_counts else activity_count
        t = np.log1p(time_since_last_activity)
        return np.array([a, submission, extension, tutor_contact,
                         future_registration, t], dtype=float)

    @property
    def n_features(self) -> int:
        return 6

    def standardize(self, X: np.ndarray, fit: bool = False):
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-8
            return (X - self.mean_) / self.std_
        return (X - self.mean_) / self.std_