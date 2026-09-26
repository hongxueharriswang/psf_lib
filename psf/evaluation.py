"""Evaluation metrics for persistence models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def _safe_pr_auc(y_true, y_score):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return np.nan

    return average_precision_score(y_true, y_score)


def _safe_brier(y_true, y_prob):
    from sklearn.metrics import brier_score_loss
    if len(np.unique(y_true)) < 2:
        return np.nan
    return brier_score_loss(y_true, y_prob)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard metrics for a binary classifier."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "auc": _safe_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "brier": _safe_brier(y_true, y_prob),
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def time_dependent_auc(
    survival_time: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
    times: np.ndarray | None = None,
) -> dict[int, float]:
    """Compute AUC at specified time points for a survival model.

    A student is a "case" at time t if they experienced the event at t;
    they are a "control" if they survived past t (with appropriate
    censoring adjustment). This is a simplified version using
    cumulative/dynamic definitions.
    """
    if times is None:
        times = np.unique(survival_time[event == 1])
    out = {}
    for t in times:
        cases = (survival_time == t) & (event == 1)
        controls = (survival_time > t) | ((survival_time == t) & (event == 0))
        if cases.sum() < 1 or controls.sum() < 1:
            out[int(t)] = np.nan
            continue
        scores = np.concatenate([risk_scores[cases], risk_scores[controls]])
        labels = np.concatenate([np.ones(cases.sum()), np.zeros(controls.sum())])
        out[int(t)] = _safe_auc(labels, scores)
    return out


def concordance_index(
    survival_time: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Harrell's concordance index for right-censored data."""
    n = len(survival_time)
    concordant = 0
    discordant = 0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            if survival_time[i] < survival_time[j] and event[i] == 1:
                comparable += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
            elif survival_time[j] < survival_time[i] and event[j] == 1:
                comparable += 1
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] < risk_scores[i]:
                    discordant += 1
    if comparable == 0:
        return np.nan
    return (concordant + 0.5 * (comparable - concordant - discordant)) / comparable


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a tidy comparison table from a dict of metric dicts."""
    import pandas as pd
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")

def evaluate_at_positive_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    positive_rate: float,
) -> dict[str, float]:
    """Evaluate a classifier by flagging the top `positive_rate` fraction.

    This is the operationally relevant regime for early-warning systems:
    advisors can contact only a fixed number of students per week, so the
    question is "of the students we flag, how many are true withdrawals?"

    Parameters
    ----------
    y_true : (N,) int array of 0/1 labels
    y_score : (N,) float risk scores (higher = more likely to withdraw)
    positive_rate : float in (0, 1), fraction of students to flag

    Returns
    -------
    dict with keys: auc, pr_auc, precision, recall, f1, tpr, fpr, ...
    """
    n = len(y_true)
    k = max(1, int(round(positive_rate * n)))
    order = np.argsort(-y_score)
    y_pred = np.zeros(n, dtype=int)
    y_pred[order[:k]] = 1

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)

    return {
        "auc": _safe_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "brier": _safe_brier(y_true, y_score),
        "positive_rate": positive_rate,
        "k": k,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": recall,
        "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
