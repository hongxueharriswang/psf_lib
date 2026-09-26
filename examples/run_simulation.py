"""End-to-end simulation study for the PSF.

This script:
1. Builds a "true" PSFModel with known parameters.
2. Simulates a cohort of students.
3. Re-fits the model via Baum-Welch.
4. Compares the fitted model to the true model.
5. Compares PSF-based withdrawal predictions to three baselines.
"""
import numpy as np

from ..psf import (
    InactivityThresholdClassifier,
    LogisticDropoutModel,
    PSFModel,
    SurvivalWithoutIntent,
    baum_welch,
    compare_models,
    evaluate_predictions,
    forward_backward,
    simulate_cohort,
)
from ..psf.states import IDX


def build_true_model(n_features: int = 6) -> PSFModel:
    m = PSFModel(n_features=n_features)

    # Transition logits (rows = from-state)
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
    m.A_logits = A_logits
    m._refresh_A()

    # Emission means/variances (distinct per state)
    rng = np.random.default_rng(42)
    m.mu = rng.normal(loc=0, scale=1, size=(6, n_features))
    m.mu[IDX.S2] += np.array([1.0, 0.8, 0.0, 0.0, 0.0, -0.5])
    m.mu[IDX.S3_NO_INTENT] += np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 1.0])
    m.mu[IDX.S3_INTENT] += np.array([-0.5, -0.3, 0.5, 0.5, 0.8, 0.8])
    m.mu[IDX.S4] += np.array([-2.0, -1.0, -0.5, -0.5, -0.5, 1.5])
    m.mu[IDX.S5] += np.array([0.5, 1.2, 0.2, 0.0, 0.0, -1.0])
    m.log_sigma = np.log(np.full((6, n_features), 0.7))
    return m


def main():
    print("=== PSF Simulation Study ===\n")

    true_model = build_true_model()
    print("True model:")
    print(true_model)

    # 1. Simulate cohort
    print("\n[1] Simulating cohort...")
    data = simulate_cohort(true_model, n_students=400, max_time=36, missing_rate=0.2)
    obs = data["observations"]
    masks = data["masks"]
    print(f"    Simulated {len(obs)} students; "
          f"mean duration = {data['durations'].mean():.1f}")

    # 2. Fit model via EM
    print("\n[2] Fitting model via Baum-Welch...")
    init_model = PSFModel(n_features=true_model.n_features)
    rng = np.random.default_rng(0)
    init_model.mu = rng.normal(size=init_model.mu.shape) * 0.5
    init_model.log_sigma = np.zeros_like(init_model.log_sigma)
    fitted = baum_welch(init_model, obs, masks, n_iter=30, verbose=True)

    # 3. Compare posteriors to true states
    print("\n[3] Evaluating state-recovery accuracy...")
    correct, total = 0, 0
    for seq, m, true_states in zip(obs, masks, data["states"]):
        out = forward_backward(fitted, seq, m, return_xi=False)
        pred = out["gamma"].argmax(axis=1)
        correct += int((pred == true_states).sum())
        total += len(true_states)
    print(f"    Joint-state accuracy: {correct / total:.3f}")

    # 4. Compare to baselines on withdrawal prediction
    print("\n[4] Comparing to baselines...")
    labels = data["final_label"]
    y_withdraw = (labels == 1).astype(int)
    # Features: mean observation over each student's trajectory
    X = np.array([o.mean(axis=0) for o in obs])
    # Inactivity proxy: last feature = log1p(time since last activity)
    inactivity = X[:, 5]

    # PSF-based withdrawal probability at final timestep
    psf_prob = []
    for seq, m in zip(obs, masks):
        out = forward_backward(fitted, seq, m, return_xi=False)
        g = out["gamma"][-1]
        psf_prob.append(g[IDX.S4] + g[IDX.S3_NO_INTENT])
    psf_prob = np.asarray(psf_prob)

    results = {}
    results["PSF (fitted)"] = evaluate_predictions(y_withdraw, psf_prob)
    results["Inactivity threshold"] = evaluate_predictions(
        y_withdraw,
        InactivityThresholdClassifier(threshold=1.5).predict_proba(inactivity)[:, 1],
    )
    logit = LogisticDropoutModel().fit(X, y_withdraw)
    results["Logistic regression"] = evaluate_predictions(
        y_withdraw, logit.predict_proba(X)[:, 1]
    )
    surv = SurvivalWithoutIntent().fit(X, data["durations"], labels)
    cif = surv.predict_cumulative_incidence(X, horizon=12)
    results["Survival (no intent)"] = evaluate_predictions(
        y_withdraw, cif[:, -1, 1]
    )

    table = compare_models(results)
    print("\n" + table.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()