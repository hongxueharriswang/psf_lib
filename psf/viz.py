"""Visualization utilities for the Persistence-State Framework.

All functions return the matplotlib Figure they create, so callers can
further customize or save them. Set ``save_path`` to write to disk.

matplotlib is imported lazily, so it is not a hard dependency of the
core library. Install the optional extra:

    pip install psf-learning[viz]
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .states import IDX, JOINT_STATE_LABELS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_mpl():
    """Import matplotlib with a non-interactive backend and return pyplot."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _save_or_show(fig, save_path: Optional[str]):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_em_convergence(
    histories: Dict[str, Sequence[float]],
    save_path: Optional[str] = None,
):
    """Plot EM log-likelihood trajectories (one line per restart).

    Parameters
    ----------
    histories : dict
        Maps a label (e.g., ``"restart 0"``) to a list of per-iteration
        log-likelihoods.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for label, hist in histories.items():
        ax.plot(range(1, len(hist) + 1), hist, marker="o",
                markersize=3, linewidth=1.2, label=label)

    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Total log-likelihood")
    ax.set_title("EM convergence across restarts")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_restart_scores(
    scores: Dict[str, float],
    save_path: Optional[str] = None,
):
    """Bar chart of final log-likelihood per restart, highlighting the best."""
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    labels = list(scores.keys())
    values = [scores[k] for k in labels]
    best = max(values)
    colors = ["#1A73E8" if v == best else "#B0BEC5" for v in values]

    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Final log-likelihood")
    ax.set_title("Restart comparison (best highlighted)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_transition_matrix(
    A: np.ndarray,
    title: str = "Transition matrix",
    save_path: Optional[str] = None,
    annotate: bool = True,
):
    """Heatmap of a transition matrix (rows = from-state)."""
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(A.shape[0]))
    ax.set_yticks(range(A.shape[0]))
    ax.set_xticklabels(JOINT_STATE_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(JOINT_STATE_LABELS, fontsize=8)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title)

    if annotate:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                if v > 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v > 0.5 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_transition_comparison(
    A_true: np.ndarray,
    A_fit: np.ndarray,
    save_path: Optional[str] = None,
):
    """Side-by-side heatmaps of true and fitted transition matrices."""
    plt = _import_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    im = None
    for ax, A, label in zip(
        axes,
        [A_true, A_fit],
        ["True transition matrix", "Fitted transition matrix (aligned)"],
    ):
        im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(A.shape[0]))
        ax.set_yticks(range(A.shape[0]))
        ax.set_xticklabels(JOINT_STATE_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(JOINT_STATE_LABELS, fontsize=7)
        ax.set_title(label, fontsize=10)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                if v > 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if v > 0.5 else "black")

    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.023, pad=0.02,
                     label="Probability")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_emission_means(
    mu_true: np.ndarray,
    mu_fit: np.ndarray,
    feature_names: Sequence[str],
    save_path: Optional[str] = None,
):
    """Grid of grouped bar charts: true vs fitted emission means per state."""
    plt = _import_mpl()
    K, D = mu_true.shape
    n_cols = 3
    n_rows = int(np.ceil(K / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.0 * n_rows),
                             sharey=True)
    axes = np.atleast_2d(axes).ravel()

    x = np.arange(D)
    width = 0.38

    for k in range(K):
        ax = axes[k]
        ax.bar(x - width / 2, mu_true[k], width,
               label="True", color="#1A73E8", edgecolor="black",
               linewidth=0.5)
        ax.bar(x + width / 2, mu_fit[k], width,
               label="Fitted", color="#FB8C00", edgecolor="black",
               linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(JOINT_STATE_LABELS[k], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)

    for k in range(K, len(axes)):
        axes[k].axis("off")

    fig.suptitle("Emission means: true vs fitted", y=1.02, fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_state_trajectory(
    gamma: np.ndarray,
    title: str = "State posterior trajectory",
    true_states: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """Stacked-area plot of joint-state posteriors over time."""
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(10.0, 4.0))

    T, K = gamma.shape
    t = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    ax.stackplot(t, gamma.T, labels=JOINT_STATE_LABELS,
                 colors=colors, alpha=0.85)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Posterior probability")
    ax.set_title(title)

    if true_states is not None:
        y_true = 1.02 + 0.06 * (np.asarray(true_states) / max(K - 1, 1))
        ax.plot(t, y_true, color="black", linewidth=1.0,
                drawstyle="steps-post", label="True state (rescaled)")
        ax.set_ylim(0, 1.15)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_intent_trajectory(
    intent_probs: np.ndarray,
    title: str = "Intent-to-Return posterior",
    true_intent: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """Line plot of P(I_t = 1 | y_{1:T}) over time."""
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(9.0, 3.5))

    T = len(intent_probs)
    t = np.arange(T)

    ax.plot(t, intent_probs, marker="o", markersize=4,
            linewidth=1.5, color="#1A73E8", label="P(I=1 | history)")

    if true_intent is not None:
        ax.scatter(t, true_intent, marker="x", s=40, color="black",
                   label="True I_t", zorder=5)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time step")
    ax.set_ylabel("P(I_t = 1)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray,
    scores: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """ROC and precision-recall curves for multiple models."""
    from sklearn.metrics import (
        roc_curve, precision_recall_curve,
        roc_auc_score, average_precision_score,
    )

    plt = _import_mpl()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(scores)))

    for (name, s), color in zip(scores.items(), colors):
        s = np.asarray(s)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, s)
        ax_roc.plot(fpr, tpr, linewidth=1.5, color=color,
                    label=f"{name} (AUC={roc_auc_score(y_true, s):.3f})")

        prec, rec, _ = precision_recall_curve(y_true, s)
        ap = average_precision_score(y_true, s)
        ax_pr.plot(rec, prec, linewidth=1.5, color=color,
                   label=f"{name} (AP={ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray",
                linewidth=0.8, label="Chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curves")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(fontsize=8, loc="lower right")

    baseline = float(np.mean(y_true))
    ax_pr.axhline(baseline, linestyle="--", color="gray", linewidth=0.8,
                  label=f"Chance (AP={baseline:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall curves")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_confusion_matrices(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
):
    """Grid of confusion matrices for several models."""
    plt = _import_mpl()
    models = list(results.keys())
    n = len(models)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 3.3 * n_rows))
    axes = np.atleast_2d(axes).ravel()

    for ax, name in zip(axes, models):
        r = results[name]
        cm = np.array([[r["tn"], r["fp"]],
                       [r["fn"], r["tp"]]])
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=8)
        ax.set_title(name, fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=11,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Confusion matrices at default 0.5 threshold", y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_metric_bar_chart(
    results: Dict[str, Dict[str, float]],
    metrics: Sequence[str] = ("auc", "pr_auc", "brier"),
    save_path: Optional[str] = None,
):
    """Grouped bar chart comparing models across selected metrics."""
    plt = _import_mpl()
    models = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 3.8))
    axes = np.atleast_1d(axes)

    x = np.arange(len(models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for ax, metric in zip(axes, metrics):
        vals = [results[m].get(metric, np.nan) for m in models]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
        ax.set_title(metric.upper())
        ax.grid(True, axis="y", alpha=0.3)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7)

    fig.suptitle("Model comparison", y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_withdrawal_curves(
    model,
    gamma: np.ndarray,
    horizon: int = 12,
    save_path: Optional[str] = None,
):
    """Plot P(Withdraw within h) as a function of h for a single student."""
    plt = _import_mpl()
    K = model.n_states

    A_h = np.eye(K)
    p_withdraw = np.zeros(horizon)
    for h in range(1, horizon + 1):
        A_h = A_h @ model.A
        p_withdraw[h - 1] = gamma[-1] @ A_h[:, IDX.S4]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(range(1, horizon + 1), p_withdraw,
            marker="o", linewidth=1.5, color="#C62828")
    ax.set_xlabel("Horizon (time steps ahead)")
    ax.set_ylabel("P(Withdrawn within horizon)")
    ax.set_title("Cumulative withdrawal probability from the last observed state")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig