"""End-to-end simulation study for the Persistence-State Framework (PSF).

This script provides a self-contained verification of the PSF inference
machinery on synthetic data. Its purpose is to confirm that:

1. Baum-Welch (EM) converges monotonically.
2. The fitted model recovers the true latent structure (state recovery,
   parameter recovery).
3. Missing data is handled correctly.
4. The Intent-to-Return variable is recoverable from proxies.

It also compares the PSF to three baselines (inactivity threshold,
logistic regression, survival without intent) on withdrawal prediction
and produces a set of diagnostic figures.

Important framing
-----------------
The simulation's label is a deterministic function of the terminal joint
state, and that state is directly observable from the last value of the
inactivity feature. The inactivity baseline is therefore near-Bayes-optimal
by construction. The simulation's role is *machine verification*, not
predictive superiority. The baseline comparison is provided for
completeness and to expose the design property; substantive comparison
belongs on real institutional data.

Run:
    python examples/01_simulation_study.py

Figures are saved to ./figures/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from psf import (
    PSFModel,
    baum_welch,
    forward_backward,
    viterbi,
    simulate_cohort,
    InactivityThresholdClassifier,
    LogisticDropoutModel,
    SurvivalWithoutIntent,
    evaluate_predictions,
    compare_models,
)
from psf.states import IDX, JOINT_STATE_LABELS
from psf.evaluation import evaluate_at_positive_rate
from psf import viz


FIG_DIR = Path("figures")


# ---------------------------------------------------------------------------
# True model specification
# ---------------------------------------------------------------------------

def build_true_model(n_features: int = 6) -> PSFModel:
    """Build a synthetic 'true' PSFModel with known parameters."""
    model = PSFModel(n_features=n_features)

    A_logits = np.full((6, 6), -np.inf)
    A_logits[IDX.S1, IDX.S1] = 1.0
    A_logits[IDX.S1, IDX.S2] = 3.0
    A_logits[IDX.S2, IDX.S2] = 3.0
    A_logits[IDX.S2, IDX.S3_NO_INTENT] = 0.0
    A_logits[IDX.S2, IDX.S3_INTENT] = 1.0
    A_logits[IDX.S2, IDX.S4] = -2.0
    A_logits[IDX.S2, IDX.S5] = -0.5
    A_logits[IDX.S3_NO_INTENT, IDX.S3_NO_INTENT] = 2.0
    A_logits[IDX.S3_NO_INTENT, IDX.S3_INTENT] = 0.0
    A_logits[IDX.S3_NO_INTENT, IDX.S4] = 0.0
    A_logits[IDX.S3_INTENT, IDX.S3_INTENT] = 1.0
    A_logits[IDX.S3_INTENT, IDX.S2] = 0.5
    A_logits[IDX.S3_INTENT, IDX.S4] = -1.0
    A_logits[IDX.S4, IDX.S4] = 0.0
    A_logits[IDX.S5, IDX.S5] = 0.0
    model.A_logits = A_logits
    model._refresh_A()

    rng = np.random.default_rng(42)
    model.mu = rng.normal(loc=0.0, scale=1.0, size=(6, n_features))
    model.mu[IDX.S2] += np.array([1.0, 0.8, 0.0, 0.0, 0.0, -0.5])
    model.mu[IDX.S3_NO_INTENT] += np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 1.0])
    model.mu[IDX.S3_INTENT] += np.array([-0.5, -0.3, 0.5, 0.5, 0.8, 0.8])
    model.mu[IDX.S4] += np.array([-2.0, -1.0, -0.5, -0.5, -0.5, 1.5])
    model.mu[IDX.S5] += np.array([0.5, 1.2, 0.2, 0.0, 0.0, -1.0])
    model.log_sigma = np.log(np.full((6, n_features), 0.7))
    return model


# ---------------------------------------------------------------------------
# Multi-restart EM
# ---------------------------------------------------------------------------

def fit_with_restarts(
    observations,
    masks,
    n_restarts: int = 5,
    n_iter: int = 30,
) -> tuple[PSFModel, dict]:
    """Fit a PSFModel via Baum-Welch with multiple random restarts.

    Returns
    -------
    best_model : PSFModel
    histories : dict
        Maps ``"restart k"`` to the EM log-likelihood history.
    """
    n_features = observations[0].shape[1]
    best_model, best_ll, best_seed = None, -np.inf, -1
    histories: dict = {}
    final_lls: dict = {}

    for seed in range(n_restarts):
        rng = np.random.default_rng(seed)
        model = PSFModel(n_features=n_features)
        model.mu = rng.normal(size=model.mu.shape) * 0.5
        model.log_sigma = np.zeros_like(model.log_sigma)

        baum_welch(model, observations, masks, n_iter=n_iter, verbose=False)

        ll = sum(forward_backward(model, o, m)["log_lik"]
                 for o, m in zip(observations, masks))

        label = f"restart {seed}"
        histories[label] = list(getattr(model, "em_history_", []))
        final_lls[label] = ll

        marker = ""
        if ll > best_ll:
            best_model, best_ll, best_seed = model, ll, seed
            marker = "  <-- new best"
        print(f"    {label}: final LL = {ll:12.2f}{marker}")

    print(f"    selected restart: seed={best_seed}, LL={best_ll:.2f}")
    return best_model, {"histories": histories, "final_lls": final_lls}


# ---------------------------------------------------------------------------
# Parameter-recovery diagnostics
# ---------------------------------------------------------------------------

def align_states(true_model: PSFModel, fitted_model: PSFModel) -> np.ndarray:
    """Find the fitted-state permutation that best matches true states.

    Returns
    -------
    perm : (K,) int array
        ``perm[j]`` is the true-state index assigned to fitted state ``j``.
    """
    K = true_model.n_states
    cost_mu = np.linalg.norm(
        true_model.mu[:, None, :] - fitted_model.mu[None, :, :], axis=2
    )
    cost_A = np.linalg.norm(
        true_model.A[:, None, :] - fitted_model.A[None, :, :], axis=2
    )
    cost = cost_mu + cost_A
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros(K, dtype=int)
    perm[col_ind] = row_ind
    return perm


def print_parameter_recovery(
    true_model: PSFModel,
    fitted_model: PSFModel,
) -> np.ndarray:
    """Compare fitted parameters to true parameters after state alignment."""
    perm = align_states(true_model, fitted_model)
    print(f"  State alignment (fitted j -> true): {perm.tolist()}")

    A_fit_aligned = fitted_model.A[perm][:, perm]
    mu_fit_aligned = fitted_model.mu[perm]

    print("\n  --- Transition matrix: TRUE ---")
    print(pd.DataFrame(
        true_model.A, index=JOINT_STATE_LABELS, columns=JOINT_STATE_LABELS,
    ).round(3).to_string())

    print("\n  --- Transition matrix: FITTED (aligned) ---")
    print(pd.DataFrame(
        A_fit_aligned, index=JOINT_STATE_LABELS, columns=JOINT_STATE_LABELS,
    ).round(3).to_string())

    feat_names = ["activity", "submission", "extension",
                  "tutor", "future_reg", "inactivity"]
    mean_diff = np.abs(true_model.mu - mu_fit_aligned)
    print(f"\n  Emission mean absolute difference: "
          f"mean={mean_diff.mean():.3f}, max={mean_diff.max():.3f}")
    print(pd.DataFrame(
        mean_diff, index=JOINT_STATE_LABELS, columns=feat_names,
    ).round(3).to_string())

    return perm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=== PSF Simulation Study ===\n")

    # ---- 1. Build true model ----
    true_model = build_true_model()
    print(f"True model: {true_model}")

    # ---- 2. Simulate cohort ----
    print("\n[1] Simulating cohort...")
    data = simulate_cohort(
        true_model, n_students=400, max_time=36, missing_rate=0.2,
    )
    print(f"    Simulated {len(data['observations'])} students; "
          f"mean duration = {data['durations'].mean():.1f}")

    # ---- 3. Fit via EM with restarts ----
    print("\n[2] Fitting model via Baum-Welch (multi-restart)...")
    fitted, fit_info = fit_with_restarts(
        data["observations"], data["masks"], n_restarts=5, n_iter=30,
    )

    # Figure: EM convergence across restarts
    viz.plot_em_convergence(
        fit_info["histories"],
        save_path=str(FIG_DIR / "01_em_convergence.png"),
    )
    viz.plot_restart_scores(
        fit_info["final_lls"],
        save_path=str(FIG_DIR / "02_restart_scores.png"),
    )

    # ---- 4. Parameter recovery ----
    print("\n[3] Parameter recovery diagnostic:")
    perm = print_parameter_recovery(true_model, fitted)

    # Figures: transition and emission comparison
    A_fit_aligned = fitted.A[perm][:, perm]
    mu_fit_aligned = fitted.mu[perm]
    feat_names = ["activity", "submission", "extension",
                  "tutor", "future_reg", "inactivity"]

    viz.plot_transition_comparison(
        true_model.A, A_fit_aligned,
        save_path=str(FIG_DIR / "03_transition_matrices.png"),
    )
    viz.plot_emission_means(
        true_model.mu, mu_fit_aligned, feat_names,
        save_path=str(FIG_DIR / "04_emission_means.png"),
    )

    # ---- 5. State recovery ----
    print("\n[4] Evaluating state-recovery accuracy...")
    correct_post, correct_vit, total = 0, 0, 0
    for seq, m, true_states in zip(
        data["observations"], data["masks"], data["states"]
    ):
        out = forward_backward(fitted, seq, m)
        pred_post = out["gamma"].argmax(axis=1)
        pred_vit = viterbi(fitted, seq, m)
        correct_post += int((pred_post == true_states).sum())
        correct_vit += int((pred_vit == true_states).sum())
        total += len(true_states)
    print(f"    Posterior-argmax accuracy: {correct_post / total:.3f}")
    print(f"    Viterbi accuracy:          {correct_vit / total:.3f}")

    # Figure: state trajectories for the first 4 test-like students
    for i in range(min(4, len(data["observations"]))):
        seq = data["observations"][i]
        m = data["masks"][i]
        true_states = data["states"][i]
        out = forward_backward(fitted, seq, m)
        viz.plot_state_trajectory(
            out["gamma"],
            title=f"Student {i}: joint-state posterior",
            true_states=true_states,
            save_path=str(FIG_DIR / f"05_state_trajectory_student_{i}.png"),
        )

    # Figure: Intent trajectory for a student who stopped out and returned
    from psf import intent_posterior
    for i in range(min(20, len(data["observations"]))):
        seq = data["observations"][i]
        m = data["masks"][i]
        true_states = data["states"][i]
        s3_mask = (true_states == IDX.S3_INTENT) | (true_states == IDX.S3_NO_INTENT)
        if s3_mask.sum() >= 3:
            out = forward_backward(fitted, seq, m)
            intent = intent_posterior(out["gamma"])
            true_intent = (true_states == IDX.S3_INTENT).astype(float)
            viz.plot_intent_trajectory(
                intent,
                title=f"Student {i}: Intent-to-Return posterior",
                true_intent=true_intent,
                save_path=str(FIG_DIR / f"06_intent_student_{i}.png"),
            )
            break  # one example is enough

    # ---- 6. Baseline comparison ----
    print("\n[5] Comparing to baselines...")
    labels = data["final_label"]
    y_withdraw = (labels == 1).astype(int)
    X = np.array([o.mean(axis=0) for o in data["observations"]])
    inactivity = np.array([o[-1, 5] for o in data["observations"]])

    psf_prob = []
    psf_prob_viterbi = []
    for seq, m in zip(data["observations"], data["masks"]):
        out = forward_backward(fitted, seq, m, return_xi=False)
        g = out["gamma"]
        psf_prob.append(float(g[-1, IDX.S4] + g[-1, IDX.S3_NO_INTENT]))
        path = viterbi(fitted, seq, m)
        psf_prob_viterbi.append(
            1.0 if path[-1] in (IDX.S4, IDX.S3_NO_INTENT) else 0.0
        )
    psf_prob = np.asarray(psf_prob)
    psf_prob_viterbi = np.asarray(psf_prob_viterbi)

    # Align simulation labels with SurvivalWithoutIntent's event codes
    event_for_survival = np.zeros_like(labels)
    event_for_survival[labels == 0] = 3
    event_for_survival[labels == 1] = 2
    event_for_survival[labels == 2] = 0
    event_for_survival[labels == 3] = 0

    inactivity_score = (
        InactivityThresholdClassifier(threshold=1.0)
        .predict_proba(inactivity)[:, 1]
    )
    logit = LogisticDropoutModel().fit(X, y_withdraw)
    logit_score = logit.predict_proba(X)[:, 1]
    surv = SurvivalWithoutIntent().fit(X, data["durations"], event_for_survival)
    cif = surv.predict_cumulative_incidence(X, horizon=12)
    surv_score = cif[:, -1, 1]

    scores = {
        "PSF (posterior)": psf_prob,
        "PSF (Viterbi)": psf_prob_viterbi,
        "Inactivity threshold": inactivity_score,
        "Logistic regression": logit_score,
        "Survival (no intent)": surv_score,
    }

    results = {name: evaluate_predictions(y_withdraw, s)
               for name, s in scores.items()}
    table = compare_models(results)
    print("\n" + table.to_string(float_format=lambda x: f"{x:.3f}"))

    # Figures: ROC/PR, confusion matrices, metric bars
    viz.plot_roc_pr_curves(
        y_withdraw, scores,
        save_path=str(FIG_DIR / "07_roc_pr_curves.png"),
    )
    viz.plot_confusion_matrices(
        results,
        save_path=str(FIG_DIR / "08_confusion_matrices.png"),
    )
    viz.plot_metric_bar_chart(
        results, metrics=("auc", "pr_auc", "f1"),
        save_path=str(FIG_DIR / "09_metric_bars.png"),
    )

    # ---- 7. Matched-rate comparison ----
    prevalence = float(y_withdraw.mean())
    print(f"\n[6] Matched-rate comparison at top {prevalence:.1%} of students...")
    matched = {
        name: evaluate_at_positive_rate(y_withdraw, s, prevalence)
        for name, s in scores.items()
    }
    matched_table = compare_models(matched)
    print("\n" + matched_table.to_string(float_format=lambda x: f"{x:.3f}"))

    # Figure: precision at matched rate
    fig_metrics = {
        name: {"auc": m["auc"], "pr_auc": m["pr_auc"], "precision": m["precision"]}
        for name, m in matched.items()
    }
    viz.plot_metric_bar_chart(
        fig_metrics, metrics=("precision", "auc", "pr_auc"),
        save_path=str(FIG_DIR / "10_matched_rate_metrics.png"),
    )

    # ---- 8. Cumulative withdrawal curve for one student ----
    print("\n[7] Cumulative withdrawal probability for one student...")
    example_idx = int(np.argmax(psf_prob))
    seq = data["observations"][example_idx]
    m = data["masks"][example_idx]
    out = forward_backward(fitted, seq, m)
    viz.plot_withdrawal_curves(
        fitted, out["gamma"], horizon=12,
        save_path=str(FIG_DIR / "11_withdrawal_curve.png"),
    )

    # ---- 9. Interpretation notes ----
    print(f"\nAll figures saved to {FIG_DIR.resolve()}")
    print("\nInterpretation notes:")
    print("  - The simulation label is a deterministic function of the")
    print("    terminal joint state, and the terminal state is directly")
    print("    observable from the final value of the inactivity feature.")
    print("  - The inactivity baseline is therefore near-Bayes-optimal by")
    print("    construction; its strong performance is a design property,")
    print("    not a failure of the PSF.")
    print("  - The PSF's role in this simulation is to verify that the EM")
    print("    and forward-backward machinery recover the true latent")
    print("    structure. See sections [3] and [4] for that verification.")
    print("  - The substantive baseline comparison belongs on real")
    print("    institutional data where the terminal state is genuinely")
    print("    latent and offline learning is not directly observable.")


if __name__ == "__main__":
    main()