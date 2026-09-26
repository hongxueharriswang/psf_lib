"""PSF: Persistence-State Framework for self-paced learning.

A Python library for empirical validation of the framework described in
"Distinguishing Stop-Out from Attrition in Self-Paced Online Learning:
A Computational Persistence Framework."
"""
from . import viz  # noqa: F401
from .baselines import (
    InactivityThresholdClassifier,
    LogisticDropoutModel,
    SurvivalWithoutIntent,
)
from .evaluation import (
    compare_models,
    concordance_index,
    evaluate_predictions,
    time_dependent_auc,
)
from .features import FeatureExtractor
from .inference import (
    forward_backward,
    particle_filter,
    return_probabilities,
    viterbi,
    withdrawal_probabilities,
)
from .labels import (
    StudentRecord,
    build_training_examples,
    generate_operational_labels,
)
from .learning import baum_welch
from .model import PSFModel
from .simulation import simulate_cohort
from .states import (
    JOINT_STATE_LABELS,
    JointState,
    PersistenceState,
    collapse_posterior,
    intent_posterior,
)
from .survival import CompetingRiskSurvival

__version__ = "0.1.0"

__all__ = [
    "JOINT_STATE_LABELS",
    "CompetingRiskSurvival",
    "FeatureExtractor",
    "InactivityThresholdClassifier",
    "JointState",
    "LogisticDropoutModel",
    "PSFModel",
    "PersistenceState",
    "StudentRecord",
    "SurvivalWithoutIntent",
    "baum_welch",
    "build_training_examples",
    "collapse_posterior",
    "compare_models",
    "concordance_index",
    "evaluate_predictions",
    "forward_backward",
    "generate_operational_labels",
    "intent_posterior",
    "particle_filter",
    "return_probabilities",
    "simulate_cohort",
    "time_dependent_auc",
    "viterbi",
    "withdrawal_probabilities",
]