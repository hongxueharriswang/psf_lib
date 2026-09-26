"""Tests for operational labels."""
import numpy as np
from psf import StudentRecord, generate_operational_labels


def test_completed_student():
    r = StudentRecord(
        student_id=1,
        registration_times=[0],
        last_activity_time=10,
        completion_time=10,
    )
    labels = generate_operational_labels([r], W=24, T_pause=2, observation_end=36)
    assert labels[0] == 0


def test_formal_withdrawal():
    r = StudentRecord(
        student_id=2,
        registration_times=[0],
        last_activity_time=5,
        withdrawal_time=6,
    )
    labels = generate_operational_labels([r], W=24, T_pause=2, observation_end=36)
    assert labels[0] == 1


def test_stopped_out_with_intent():
    r = StudentRecord(
        student_id=3,
        registration_times=[0],
        last_activity_time=5,
        extension_active_after=10,
    )
    labels = generate_operational_labels([r], W=24, T_pause=2, observation_end=20)
    assert labels[0] == 2


def test_withdrawn_without_intent():
    r = StudentRecord(
        student_id=4,
        registration_times=[0],
        last_activity_time=5,
    )
    labels = generate_operational_labels([r], W=12, T_pause=2, observation_end=36)
    assert labels[0] == 1


def test_censored_insufficient_followup():
    r = StudentRecord(
        student_id=5,
        registration_times=[0],
        last_activity_time=5,
        extension_active_after=10,
    )
    labels = generate_operational_labels([r], W=24, T_pause=2, observation_end=6)
    assert labels[0] == 3