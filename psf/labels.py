"""Operational definitions of stop-out, withdrawal, and completion.

These labels are generated independently of the PSF model and are used
as ground truth for training and evaluation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class StudentRecord:
    """Minimal per-student record for label generation.

    Attributes
    ----------
    student_id : int
    registration_times : sequence of int (time indices when registered)
    last_activity_time : int (last observed LMS or submission activity)
    completion_time : Optional[int]
    withdrawal_time : Optional[int]   (formal withdrawal record)
    extension_active_after : Optional[int]
        Latest time up to which an extension was active.
    future_registration_after : Optional[int]
        Time of the earliest future registration (after last activity).
    """
    student_id: int
    registration_times: Sequence[int]
    last_activity_time: int
    completion_time: int | None = None
    withdrawal_time: int | None = None
    extension_active_after: int | None = None
    future_registration_after: int | None = None


def generate_operational_labels(
    records: Sequence[StudentRecord],
    W: int = 24,
    T_pause: int = 2,
    observation_end: int | None = None,
) -> np.ndarray:
    """Return an int array of operational labels per student.

    Labels:
        0 = Completed
        1 = Withdrawn (operational attrition)
        2 = Stopped-Out (operational stop-out)
        3 = Still enrolled / censored
    """
    labels = np.full(len(records), 3, dtype=int)  # default: still enrolled
    for i, r in enumerate(records):
        if r.completion_time is not None:
            labels[i] = 0
            continue
        if r.withdrawal_time is not None:
            labels[i] = 1
            continue
        # Inactivity-based classification
        t_last = r.last_activity_time
        t_ext = r.extension_active_after
        t_reg = r.future_registration_after
        has_intent_signal = (t_ext is not None) or (t_reg is not None)
        obs_end = observation_end if observation_end is not None else t_last + W

        if obs_end - t_last < T_pause:
            labels[i] = 3  # insufficient follow-up
            continue

        # If returned within W or has active intent signal -> Stop-Out
        if has_intent_signal and (obs_end - t_last) <= W:
            labels[i] = 2
            continue
        # If follow-up >= W and no return and no intent signal -> Withdrawn
        if (obs_end - t_last) >= W and not has_intent_signal:
            labels[i] = 1
            continue
        # Otherwise: still stopped-out / pending
        labels[i] = 2 if has_intent_signal else 3
    return labels


def build_training_examples(
    records: Sequence[StudentRecord],
    features_by_student: np.ndarray,
    label_time: int = 6,
    W: int = 24,
    T_pause: int = 2,
) -> dict:
    """Create (X, y) training examples from student records.

    Each student contributes one example using features available at
    `label_time` (relative to last activity) and the operational label
    determined at `W` months after last activity.
    """
    labels = generate_operational_labels(records, W=W, T_pause=T_pause)
    # Keep only students whose follow-up is long enough
    keep = labels != 3
    X = features_by_student[keep]
    y = labels[keep]
    return {"X": X, "y": y, "student_ids": np.array([r.student_id for r in records])[keep]}