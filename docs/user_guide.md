# PSF User Guide

## A Practical Guide to the Persistence-State Framework for Learning Analytics

**Version 0.1.0**
**Companion to:** *Distinguishing Stop-Out from Attrition in Self-Paced Online Learning: A Computational Persistence Framework*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Installation and Setup](#3-installation-and-setup)
4. [Data Collection](#4-data-collection)
5. [Data Preparation](#5-data-preparation)
6. [Library Architecture](#6-library-architecture)
7. [Step-by-Step Validation Workflow](#7-step-by-step-validation-workflow)
8. [Understanding the Figures](#8-understanding-the-figures)
9. [Worked Examples](#9-worked-examples)
10. [The Four-Tier Validation Protocol](#10-the-four-tier-validation-protocol)
11. [Interpreting Results](#11-interpreting-results)
12. [Troubleshooting](#12-troubleshooting)
13. [Ethics and Deployment](#13-ethics-and-deployment)
14. [Extending the Library](#14-extending-the-library)
15. [Frequently Asked Questions](#15-frequently-asked-questions)
16. [References](#16-references)

---

## 1. Introduction

### 1.1 What This Guide Is For

This guide introduces the **Persistence-State Framework (PSF)** and provides practical instructions for using the accompanying Python library, `psf`, to validate the framework against institutional data. It is written for:

- **Learning analytics researchers** who want to test whether stop-out and attrition can be distinguished computationally in their own institutional context.
- **Institutional data scientists** who want to implement PSF-based early warning systems or intervention triage pipelines.
- **Graduate students** who want to replicate or extend the framework as part of a thesis or dissertation.
- **Reviewers and methodologists** who want to understand the assumptions and mechanics of the framework in detail.

### 1.2 What the PSF Is (and Is Not)

The PSF is a **conceptual and computational framework** for modeling persistence in self-paced learning. It is not a ready-to-deploy production system. It provides:

- A formal state-space representation of persistence behavior.
- A latent variable (Intent to Return) that distinguishes temporary stop-out from permanent attrition.
- A set of inference algorithms (HMM, DBN, survival analysis, particle filtering).
- A validation methodology with operational definitions and baselines.
- A set of diagnostic and visualization tools.

The PSF does **not** provide:

- Pre-trained models (these must be fitted on institutional data).
- A dashboard or user interface.
- Automated intervention delivery.
- A guarantee of external validity across institutions.

### 1.3 When to Use the PSF

Use the PSF when:

- Your institution offers self-paced or continuous-enrollment courses.
- Inactivity is a poor proxy for dropout because students regularly pause and return.
- You need to distinguish temporary stop-out from permanent withdrawal.
- You want to predict return, not just dropout.
- You have access to administrative records (registrations, extensions, tutor contacts, withdrawals).

Do **not** use the PSF when:

- Your program is strictly cohort-based with fixed term schedules.
- You only have clickstream data and no administrative records.
- You need a turnkey production system rather than a research framework.
- You need causal guarantees; the PSF is an observational model.

### 1.4 What This Version Adds

Compared to earlier drafts, this version reflects the current state of the working library:

- **Multi-restart EM fitting**, which lifted simulated AUC from 0.685 (single restart) to 0.998 (five restarts) on the reference simulation.
- **A visualization module** (`psf.viz`) with thirteen plotting functions.
- **A matched-positive-rate evaluator** (`evaluate_at_positive_rate`) for operationally meaningful model comparison.
- **A four-tier validation protocol** that separates machine verification from empirical validation.
- **Parameter-recovery diagnostics** including automatic state alignment.
- **Corrected label conventions** between simulation and survival models.

---

## 2. Theoretical Foundations

### 2.1 The Core Idea

In self-paced learning, the absence of visible activity does not necessarily mean a student has withdrawn. Many students pause for weeks or months and later return. Traditional dropout models treat all inactive students as dropouts, which:

- Overestimates attrition.
- Misdirects interventions.
- Alienates adult learners who intend to return.

The PSF addresses this by representing persistence as a **dynamic state** rather than a binary outcome, and by introducing a latent **Intent to Return** variable.

### 2.2 The Five Persistence States

| State | Label | Description |
|---|---|---|
| S₁ | Registered | Enrolled but not yet progressing. |
| S₂ | Progressing | Actively completing assessments. |
| S₃ | Stopped-Out | Temporarily inactive with probability of return. |
| S₄ | Withdrawn | Effectively exited with no intent to return. |
| S₅ | Completed | Course requirements fulfilled. |

The critical distinction is between S₃ (transient) and S₄ (absorbing).

### 2.3 Intent to Return

The latent variable Iₜ ∈ {0,1} captures the student's commitment to continue. It is inferred from proxies such as:

- Extension requests
- Tutor contacts
- Future course registrations
- Communication with advising services
- Submission of partially completed work

Intent is theoretically distinct from activity. A student may be inactive but still intend to return, or active but already intend to withdraw.

### 2.4 The Joint State Space

The library represents six joint states, splitting S₃ by intent:

| Index | Joint State | Persistence | Intent |
|---|---|---|---|
| 0 | S1_REGISTERED | Registered | — |
| 1 | S2_PROGRESSING | Progressing | — |
| 2 | S3_STOPPED_OUT_NO_INTENT | Stopped-Out | 0 |
| 3 | S3_STOPPED_OUT_INTENT | Stopped-Out | 1 |
| 4 | S4_WITHDRAWN | Withdrawn | — |
| 5 | S5_COMPLETED | Completed | — |

### 2.5 What the Model Computes

- **P(Return | History)** — probability the student returns to Progressing.
- **P(Withdraw | History)** — probability the student transitions to Withdrawn.
- **P(Iₜ = 1 | History)** — posterior probability of Intent to Return.
- **Time-varying state posteriors** — P(Zₜ = k | y₁:ₜ) for each joint state.

### 2.6 Computational Formalisms

The PSF can be implemented using:

- **Hidden Markov Models (HMMs)**: discrete-time, discrete-state.
- **Dynamic Bayesian Networks (DBNs)**: HMMs with covariates and hierarchical dependencies.
- **Survival analysis**: continuous-time, competing-risk models.
- **Event history analysis**: discrete-time competing-risk models.

The `psf` library implements all four.

---

## 3. Installation and Setup

### 3.1 Requirements

- Python ≥ 3.9
- NumPy ≥ 1.22
- SciPy ≥ 1.9
- scikit-learn ≥ 1.1
- pandas ≥ 1.4
- matplotlib ≥ 3.5 (optional, for visualization)

### 3.2 Installation

**Option A: Editable install (recommended for development)**

```bash
git clone https://github.com/hongxueharriswang/psf_lib.git
cd psf_lib
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -e ".[viz,dev]"
```

**Option B: From PyPI (once published)**

```bash
pip install psf-learning
```

**Option C: Minimal install**

```bash
pip install numpy scipy scikit-learn pandas
# then copy the psf/ package into your project
```

### 3.3 Verify Installation

```python
import psf
print(psf.__version__)
# Expected: 0.1.0

from psf import PSFModel, simulate_cohort, baum_welch
from psf import viz  # requires matplotlib
print("PSF library loaded successfully.")
```

### 3.4 Run the Reference Simulation

The fastest way to verify the library works end-to-end is the simulation study:

```bash
python examples/01_simulation_study.py
```

Expected output includes EM convergence across five restarts, parameter recovery diagnostics, and a comparison table. Figures are written to `./figures/`.

### 3.5 Package Structure

```
psf/
├── states.py        # State definitions and structural constraints
├── model.py         # PSFModel (HMM parameters)
├── inference.py     # Forward-backward, Viterbi, particle filter
├── learning.py      # Baum-Welch (EM) parameter estimation
├── survival.py      # Competing-risk discrete-time survival
├── labels.py        # Operational label generation
├── baselines.py     # Baseline models for comparison
├── evaluation.py    # AUC, PR-AUC, Brier, matched-rate evaluation
├── features.py      # Feature extraction
├── simulation.py    # Synthetic data generation
├── viz.py           # Visualization utilities
└── utils.py         # Numerical utilities
```

---

## 4. Data Collection

### 4.1 Required Data Streams

The PSF requires data from five streams. Not all streams are mandatory, but more streams produce better inference.

| Stream | Examples | Purpose |
|---|---|---|
| **Registration** | Course registrations, start dates, program enrollment | Defines entry into S1 and S2 |
| **Activity** | LMS logins, page views, video plays | Defines S2 (Progressing) |
| **Assessment** | Submission times, scores, partial completions | Defines S2 and S5 |
| **Intent proxies** | Extension requests, tutor contacts, future registrations | Infers Iₜ |
| **Administrative** | Formal withdrawals, completion records | Defines S4 and S5 |

### 4.2 Minimum Viable Dataset

At a minimum, you need:

- **Registration records**: student ID, course ID, registration date.
- **Activity records**: student ID, timestamp, activity type.
- **Withdrawal records**: student ID, withdrawal date (if applicable).
- **Completion records**: student ID, completion date (if applicable).

With only these four, you can fit a basic PSF. Adding extension requests and tutor contacts significantly improves Intent-to-Return inference.

### 4.3 Recommended Additional Data

For a strong empirical study:

- **Extension request logs** with approval status.
- **Tutor contact logs** with contact type (email, phone, meeting).
- **Advising interaction logs**.
- **Future registration records** (even if subsequently dropped).
- **Offline indicators** (textbook purchases, proctored exam scheduling).

### 4.4 Data Collection Checklist

Before proceeding, confirm:

- [ ] Data are de-identified and ethically approved.
- [ ] Each student has a unique identifier.
- [ ] Timestamps are in a consistent timezone and format.
- [ ] Course start and end dates are recorded.
- [ ] Withdrawal and completion records are complete.
- [ ] Extension requests are logged.
- [ ] Tutor contacts are logged (if available).
- [ ] Observation window is at least 24 months for ground-truth labeling.

### 4.5 Ethical Considerations

Per Section 13 of this guide:

- Students should know how persistence is measured.
- Interventions should support rather than surveil.
- Data should be stored securely and de-identified.
- Model outputs should not be used punitively.

Obtain institutional review board (IRB) approval before accessing student records.

---

## 5. Data Preparation

### 5.1 Time Grid

The PSF operates on a discrete time grid. Choose:

- **Weekly grid** for early warning applications (finer resolution, more data required).
- **Monthly grid** for program evaluation (coarser resolution, more robust).

The time grid defines the resolution of state transitions. All timestamps must be mapped to grid indices.

**Recommended default:** monthly grid, with 24-month observation window, for AU learner-paced courses.

### 5.2 Student Records

Each student is represented as a `StudentRecord`:

```python
from psf import StudentRecord

record = StudentRecord(
    student_id=1001,
    registration_times=[0],           # time index of registration
    last_activity_time=8,              # time index of last LMS/submission activity
    completion_time=None,              # time index of completion (None if not completed)
    withdrawal_time=None,              # time index of formal withdrawal
    extension_active_after=10,         # latest time extension was active
    future_registration_after=None,    # time of next registration
)
```

### 5.3 Observation Sequences

Each student's observation sequence is a 2D array of shape `(T, D)`, where:

- `T` is the number of time steps observed for that student.
- `D` is the number of features per time step.

Default features (from `FeatureExtractor`):

| Index | Feature | Description |
|---|---|---|
| 0 | `log1p(activity_count)` | Log-transformed count of LMS activities |
| 1 | `submission_occurred` | 1 if assignment submitted this period |
| 2 | `extension_requested` | 1 if extension requested |
| 3 | `tutor_contact` | 1 if tutor contact occurred |
| 4 | `future_registration` | 1 if registered for future course |
| 5 | `log1p(time_since_last_activity)` | Log-transformed inactivity duration |

### 5.4 Missing Data

Self-paced data are irregularly sampled and often missing. The PSF handles missingness via a `mask` array of shape `(T, D)` where:

- `True` means the feature is observed.
- `False` means the feature is missing.

Missing features are marginalized out of the Gaussian emission. Missing timesteps are propagated forward without a state update.

### 5.5 Complete Data Preparation Example

```python
import numpy as np
from psf import FeatureExtractor, StudentRecord

def prepare_student(raw_df, student_id, time_grid="monthly"):
    """
    Convert raw event logs into a (T, D) observation matrix.
    """
    student_df = raw_df[raw_df.student_id == student_id].copy()

    # Map timestamps to time grid indices
    student_df["t"] = student_df["timestamp"].dt.month  # or .week
    T = student_df["t"].max() + 1

    extractor = FeatureExtractor()
    obs = np.zeros((T, extractor.n_features))
    mask = np.zeros((T, extractor.n_features), dtype=bool)

    for t in range(T):
        period = student_df[student_df.t == t]
        activity_count = len(period[period.event_type == "login"])
        submission = int((period.event_type == "submission").any())
        extension = int((period.event_type == "extension").any())
        tutor = int((period.event_type == "tutor_contact").any())
        future_reg = int((period.event_type == "future_registration").any())

        # Time since last activity (for feature 5)
        active = student_df[student_df.event_type.isin(["login", "submission"])]
        time_since = t - active.t.max() if len(active) > 0 else 0

        obs[t] = extractor.transform_timestep(
            activity_count, submission, extension, tutor,
            future_reg, max(time_since, 0),
        )
        mask[t] = True  # all features observed this period

    return obs, mask
```

### 5.6 Generating Operational Labels

Operational labels are generated **independently of the model** using `generate_operational_labels`:

```python
from psf import generate_operational_labels

labels = generate_operational_labels(
    records,
    W=24,           # observation window (months)
    T_pause=2,      # minimum pause to be considered Stopped-Out
    observation_end=36,
)

# Label meaning:
# 0 = Completed
# 1 = Withdrawn (operational attrition)
# 2 = Stopped-Out (operational stop-out)
# 3 = Censored / still enrolled
```

**Important:** These labels are the ground-truth target. They are defined independently of the PSF, using administrative records and observation windows. The model is trained to predict them; they are not derived from the model.

### 5.7 The Ground-Truth Definition

Two definitions are central to the validation protocol.

**Definition 1 (Operational Attrition).** A student is classified as **Withdrawn** if either:
(a) a formal withdrawal record exists, or
(b) the student has not returned to active progress within observation window W (default 24 months), and no formal extension or future registration is active.

**Definition 2 (Operational Stop-Out).** A student is classified as **Stopped-Out** if:
(a) the student has been inactive for at least T_pause (default 2 months),
(b) no formal withdrawal exists, and
(c) either the student returns within W, or has an active extension, future registration, or documented intent to return.

The choice of W and T_pause affects the label distribution. Sensitivity analysis across W ∈ {12, 24, 36} and T_pause ∈ {1, 2, 3} is recommended.

### 5.8 Data Splitting

Use **temporal splitting** to avoid data leakage:

```python
# Sort by registration time
sorted_idx = np.argsort([r.registration_times[0] for r in records])
split = int(0.7 * len(sorted_idx))
train_idx, test_idx = sorted_idx[:split], sorted_idx[split:]

train_obs = [obs[i] for i in train_idx]
train_masks = [masks[i] for i in train_idx]
test_obs = [obs[i] for i in test_idx]
test_masks = [masks[i] for i in test_idx]
```

Do **not** split randomly. Self-paced programs have temporal drift in course design, student populations, and institutional policies. Random splits leak future information into training.

Additionally, consider **leave-one-course-out** and **leave-one-cohort-out** cross-validation for stronger generalization estimates.

---

## 6. Library Architecture

### 6.1 Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Data                                │
│  (registration, activity, assessments, intent proxies)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  features.FeatureExtractor                                  │
│  → (T, D) observation arrays + masks                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  labels.generate_operational_labels                         │
│  → Operational ground-truth labels (0/1/2/3)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  model.PSFModel                                             │
│  → HMM with 6 joint states (5 persistence + intent split)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐      ┌──────────────────────┐
│ learning.        │      │ survival.            │
│ baum_welch       │      │ CompetingRiskSurvival│
│ (EM fitting)     │      │ (competing risks)    │
└────────┬─────────┘      └──────────┬───────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│  inference.forward_backward / particle_filter               │
│  → State posteriors, Intent posteriors, return probabilities│
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  evaluation.evaluate_predictions / compare_models           │
│  → AUC, PR-AUC, Brier, FPR, concordance, matched-rate       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  viz.plot_*                                                 │
│  → Publication-ready diagnostic and comparison figures      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Key Classes

| Class | Purpose |
|---|---|
| `PSFModel` | HMM parameters (transition matrix, emissions) |
| `StudentRecord` | Per-student administrative record |
| `FeatureExtractor` | Converts raw events to observation arrays |
| `CompetingRiskSurvival` | Discrete-time competing-risk survival model |
| `InactivityThresholdClassifier` | Baseline: no activity = dropout |
| `LogisticDropoutModel` | Baseline: static logistic regression |
| `SurvivalWithoutIntent` | Baseline: survival without Iₜ |

### 6.3 Key Functions

| Function | Purpose |
|---|---|
| `baum_welch` | Fit PSFModel via EM (single restart) |
| `forward_backward` | Compute state posteriors |
| `viterbi` | Most likely state sequence |
| `particle_filter` | Online state estimation |
| `return_probabilities` | P(Return) over horizon |
| `withdrawal_probabilities` | P(Withdraw) over horizon |
| `intent_posterior` | P(Iₜ = 1 | history) |
| `generate_operational_labels` | Create ground-truth labels |
| `evaluate_predictions` | Compute AUC, PR-AUC, Brier, F1 |
| `evaluate_at_positive_rate` | Matched-rate evaluation |
| `compare_models` | Build comparison table |
| `simulate_cohort` | Generate synthetic data |

### 6.4 Key Visualization Functions

| Function | Figure |
|---|---|
| `plot_em_convergence` | Log-likelihood per EM iteration, per restart |
| `plot_restart_scores` | Final log-likelihood per restart |
| `plot_transition_matrix` | Heatmap of transition matrix |
| `plot_transition_comparison` | True vs fitted transition matrices |
| `plot_emission_means` | Per-state true vs fitted emission means |
| `plot_state_trajectory` | Stacked-area posterior for one student |
| `plot_intent_trajectory` | Intent posterior for one student |
| `plot_roc_pr_curves` | ROC and PR curves for multiple models |
| `plot_confusion_matrices` | Confusion matrices for multiple models |
| `plot_metric_bar_chart` | Bar chart of selected metrics |
| `plot_withdrawal_curves` | Cumulative P(Withdraw) over horizon |

---

## 7. Step-by-Step Validation Workflow

### Step 1: Prepare Data

```python
import numpy as np
from psf import StudentRecord, FeatureExtractor

# Load institutional data
records = load_student_records()
obs, masks = load_observation_sequences()
```

### Step 2: Generate Operational Labels

```python
from psf import generate_operational_labels

labels = generate_operational_labels(records, W=24, T_pause=2, observation_end=36)

print(f"Completed:   {(labels == 0).sum()}")
print(f"Withdrawn:   {(labels == 1).sum()}")
print(f"Stopped-Out: {(labels == 2).sum()}")
print(f"Censored:    {(labels == 3).sum()}")
```

### Step 3: Split Data Temporally

```python
sorted_idx = np.argsort([r.registration_times[0] for r in records])
split = int(0.7 * len(sorted_idx))
train_idx, test_idx = sorted_idx[:split], sorted_idx[split:]
```

### Step 4: Fit the PSF with Multi-Restart

```python
from psf import PSFModel, baum_welch, forward_backward

best_model, best_ll, best_seed = None, -np.inf, -1
for seed in range(5):
    rng = np.random.default_rng(seed)
    model = PSFModel(n_features=obs[0].shape[1])
    model.mu = rng.normal(size=model.mu.shape) * 0.5
    model.log_sigma = np.zeros_like(model.log_sigma)

    baum_welch(
        model,
        [obs[i] for i in train_idx],
        [masks[i] for i in train_idx],
        n_iter=30,
    )

    ll = sum(
        forward_backward(model, obs[i], masks[i])["log_lik"]
        for i in train_idx
    )
    print(f"restart {seed}: LL = {ll:.2f}")

    if ll > best_ll:
        best_model, best_ll, best_seed = model, ll, seed

print(f"Selected restart: seed={best_seed}, LL={best_ll:.2f}")
```

**Why multi-restart matters.** EM converges to a local optimum, and the likelihood surface for a six-state HMM has multiple basins. In the reference simulation, the spread between the best and worst restart was approximately 980 log-likelihood units. A single restart has roughly a 40% chance of landing in the lowest basin. Multi-restart fitting raised the fitted model's AUC from 0.685 (single restart) to 0.998 (five restarts). Do not skip this step.

### Step 5: Inspect the Fitted Model

```python
import pandas as pd
from psf.states import JOINT_STATE_LABELS

print("Transition matrix (rows = from-state):")
print(pd.DataFrame(
    best_model.A,
    index=JOINT_STATE_LABELS,
    columns=JOINT_STATE_LABELS,
).round(3))
```

Expected: diagonal dominance, absorbing S4 and S5, and non-zero transitions between S3(I=0) and S3(I=1).

### Step 6: Compute State and Intent Posteriors

```python
from psf import forward_backward, collapse_posterior, intent_posterior

i = test_idx[0]
out = forward_backward(best_model, obs[i], masks[i])
gamma = out["gamma"]                       # (T, 6) joint-state posterior
persistence = collapse_posterior(gamma)    # (T, 5)
intent = intent_posterior(gamma)           # (T,) P(I=1 | history) at each S3 timestep

print("Persistence posteriors (last 5 timesteps):")
print(pd.DataFrame(
    persistence[-5:],
    columns=["Registered", "Progressing", "Stopped-Out", "Withdrawn", "Completed"],
).round(3))
```

### Step 7: Compute Return and Withdrawal Probabilities

```python
from psf import return_probabilities, withdrawal_probabilities

p_return = return_probabilities(best_model, gamma, horizon=6)
p_withdraw = withdrawal_probabilities(best_model, gamma, horizon=6)

print(f"P(Return within 6 months): {p_return[-1, -1]:.3f}")
print(f"P(Withdraw within 6 months): {p_withdraw[-1, -1]:.3f}")
```

### Step 8: Evaluate Against Baselines

```python
from psf import (
    InactivityThresholdClassifier,
    LogisticDropoutModel,
    SurvivalWithoutIntent,
    evaluate_predictions,
    compare_models,
    evaluate_at_positive_rate,
)
from psf.states import IDX

y_test = (labels[test_idx] == 1).astype(int)

# PSF prediction using terminal posterior
psf_prob = []
for i in test_idx:
    out = forward_backward(best_model, obs[i], masks[i])
    g = out["gamma"]
    p_terminal = g[-1, IDX.S4] + g[-1, IDX.S3_NO_INTENT]
    psf_prob.append(float(p_terminal))
psf_prob = np.asarray(psf_prob)

# Features for baselines
X_test = np.array([obs[i].mean(axis=0) for i in test_idx])
inactivity = np.array([obs[i][-1, 5] for i in test_idx])

# Baselines
results = {}
results["PSF (fitted)"] = evaluate_predictions(y_test, psf_prob)
results["Inactivity threshold"] = evaluate_predictions(
    y_test,
    InactivityThresholdClassifier(threshold=1.0).predict_proba(inactivity)[:, 1],
)
logit = LogisticDropoutModel().fit(X_test, y_test)
results["Logistic regression"] = evaluate_predictions(
    y_test, logit.predict_proba(X_test)[:, 1]
)

# Survival with label re-encoding
event_for_survival = np.zeros_like(labels)
event_for_survival[labels == 0] = 3   # Completed -> completion
event_for_survival[labels == 1] = 2   # Withdrawn -> withdrawal
event_for_survival[labels == 2] = 0   # Stopped-Out -> censored
event_for_survival[labels == 3] = 0   # Censored -> censored

surv = SurvivalWithoutIntent().fit(
    X_test, [records[i].last_activity_time for i in test_idx],
    event_for_survival[test_idx],
)
cif = surv.predict_cumulative_incidence(X_test, horizon=12)
results["Survival (no intent)"] = evaluate_predictions(y_test, cif[:, -1, 1])

# Default-threshold comparison
table = compare_models(results)
print(table[["auc", "pr_auc", "brier", "f1"]].round(3))

# Matched-rate comparison (operationally relevant)
prevalence = float(y_test.mean())
matched = {
    name: evaluate_at_positive_rate(y_test, s, prevalence)
    for name, s in {
        "PSF (fitted)": psf_prob,
        "Inactivity threshold": InactivityThresholdClassifier(threshold=1.0).predict_proba(inactivity)[:, 1],
        "Logistic regression": logit.predict_proba(X_test)[:, 1],
        "Survival (no intent)": cif[:, -1, 1],
    }.items()
}
matched_table = compare_models(matched)
print(matched_table[["auc", "pr_auc", "precision", "recall", "f1"]].round(3))
```

**Note on the survival baseline.** The simulation's label convention (0=Completed, 1=Withdrawn, 2=Stopped-Out, 3=Censored) and the survival model's event codes (0=censored, 1=return, 2=withdrawal, 3=completion) differ. If you pass simulation labels directly to `SurvivalWithoutIntent`, you will get `tp=0, fp=0` and an AUC below 0.5. Re-encode labels as shown above.

### Step 9: Sensitivity Analysis

```python
for W in [12, 24, 36]:
    for T_pause in [1, 2, 3]:
        labels_w = generate_operational_labels(
            records, W=W, T_pause=T_pause, observation_end=36,
        )
        print(f"W={W}, T_pause={T_pause}: "
              f"Withdrawn={(labels_w == 1).sum()}, "
              f"Stopped-Out={(labels_w == 2).sum()}")
```

If conclusions change materially across these settings, report the sensitivity in the paper.

---

## 8. Understanding the Figures

The visualization module produces thirteen figures during the reference simulation. Each answers a specific diagnostic question. The following table summarizes what to look for.

| Figure | File | What to look for | Diagnostic value |
|---|---|---|---|
| EM convergence | `01_em_convergence.png` | Monotone increase in each restart; spread across restarts | Confirms EM is correct; justifies multi-restart |
| Restart comparison | `02_restart_scores.png` | Best restart highlighted; gap to runner-up | Confirms the winning basin is genuine |
| Transition matrices | `03_transition_matrices.png` | Diagonal dominance; correct absorbing states; recovered S3(I=0)→S3(I=1) | Confirms structural recovery |
| Emission means | `04_emission_means.png` | True vs fitted bar alignment per state; S3 variants distinguishable | Confirms intent can be inferred |
| State trajectories | `05_state_trajectory_student_*.png` | Sharp bands matching true state; plausible sequence | Confirms latent-state inference |
| Intent trajectory | `06_intent_student_*.png` | Posterior tracking true I_t; plausible dynamics | **Central evidence for the framework** |
| ROC/PR curves | `07_roc_pr_curves.png` | PSF curve above baselines; PR more important than ROC | Confirms predictive performance |
| Confusion matrices | `08_confusion_matrices.png` | PSF: low FP and moderate FN; baselines: more FP | Confirms error profile matches operational need |
| Metric bars | `09_metric_bars.png` | PSF wins on AUC, PR-AUC, F1 | Confirms overall performance |
| Matched-rate metrics | `10_matched_rate_metrics.png` | Precision at matched capacity; PSF highest | **Operationally relevant comparison** |
| Withdrawal curve | `11_withdrawal_curve.png` | Monotone increase; saturation level | Confirms individual-level prediction |

### 8.1 The Three Key Figures

If you only look at three figures, look at these:

1. **`06_intent_student_*.png`** — This is the central evidence for the framework. It shows that the latent Intent-to-Return variable is recoverable from observable proxies, which is the mechanism that distinguishes stop-out from attrition. If this figure shows a flat line at 0.5, the intent construct is not being identified, and the model has collapsed to a three-state HMM with extra parameters.

2. **`03_transition_matrices.png`** — This confirms that the fitted transition structure matches the true one after state alignment. If the S3(I=0) → S3(I=1) transition collapses to zero, the model is not allowing students to develop intent while stopped out, which is a structural simplification that should be reported.

3. **`10_matched_rate_metrics.png`** — This is the operationally meaningful comparison. At the matched positive rate (equal to withdrawal prevalence), precision answers the advisor's question: "of the students we contact, how many actually withdraw?" The PSF should be highest here.

### 8.2 A Note on the Inactivity Baseline

In the reference simulation, the inactivity threshold achieves AUC 0.881, which is much better than chance (0.500). This is because the simulation's label is a deterministic function of the terminal joint state, and the terminal state is directly observable from the final value of the inactivity feature.

**This is a design property of the simulation, not a failure of the PSF.** In real institutional data, the label is genuinely latent and offline learning is not directly observable, so the inactivity threshold will perform worse relative to the PSF. The simulation's role is to verify that the inference machinery recovers the true latent structure (which it does); the substantive baseline comparison belongs on real data.

---

## 9. Worked Examples

### Example 1: Simulation Study (Reference)

Run the full simulation:

```bash
python examples/01_simulation_study.py
```

The expected output on a fresh run:

```
=== PSF Simulation Study ===
[1] Simulating cohort...
[2] Fitting model via Baum-Welch (multi-restart)...
    restart 0: final LL =    -41352.95
    restart 1: final LL =    -40860.32
    restart 2: final LL =    -40857.21  <-- new best
    restart 3: final LL =    -41838.84
    restart 4: final LL =    -41437.30
    selected restart: seed=2, LL=-40857.21
[3] Parameter recovery diagnostic:
[4] Evaluating state-recovery accuracy...
    Posterior-argmax accuracy: 0.859
    Viterbi accuracy:          0.912
[5] Comparing to baselines...
[6] Matched-rate comparison at top 59.8% of students...
```

All figures are saved to `./figures/`.

### Example 2: Validating Intent Recovery

Test whether the Intent-to-Return posterior tracks the true intent label:

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from psf import (
    PSFModel, simulate_cohort, baum_welch,
    forward_backward, intent_posterior,
)
from psf.states import IDX

true_model = PSFModel(n_features=6)
data = simulate_cohort(true_model, n_students=400, max_time=36)

# Multi-restart fit
best_model, best_ll = None, -np.inf
for seed in range(5):
    rng = np.random.default_rng(seed)
    m = PSFModel(n_features=6)
    m.mu = rng.normal(size=m.mu.shape) * 0.5
    m.log_sigma = np.zeros_like(m.log_sigma)
    baum_welch(m, data["observations"], data["masks"], n_iter=30)
    ll = sum(forward_backward(m, o, mk)["log_lik"]
             for o, mk in zip(data["observations"], data["masks"]))
    if ll > best_ll:
        best_model, best_ll = m, ll

# Collect intent posteriors at S3 timesteps
all_intent, all_true = [], []
for seq, m, true_states in zip(
    data["observations"], data["masks"], data["states"]
):
    out = forward_backward(best_model, seq, m)
    intent = intent_posterior(out["gamma"])
    true_intent = (true_states == IDX.S3_INTENT).astype(float)
    s3_mask = (true_states == IDX.S3_NO_INTENT) | (true_states == IDX.S3_INTENT)
    all_intent.extend(intent[s3_mask])
    all_true.extend(true_intent[s3_mask])

all_intent = np.nan_to_num(all_intent)
auc = roc_auc_score(all_true, all_intent)
print(f"Intent-to-Return AUC: {auc:.3f}")
# Expected: > 0.95 with multi-restart
```

This is the metric to report in the empirical study. An AUC above 0.95 on real data would be strong evidence that the intent construct is identified.

### Example 3: Handling Missing Data

The PSF handles missingness via masks. To test robustness, simulate with high missingness:

```python
data = simulate_cohort(true_model, n_students=400, missing_rate=0.4)

best_model, _ = None, -np.inf
for seed in range(5):
    # ... as above ...
    pass

# State recovery should remain reasonable even at 40% missingness
```

The forward-backward algorithm marginalizes missing features automatically, so accuracy is robust up to 40% missingness.

### Example 4: Online Tracking with Particle Filter

For real-time applications, use the particle filter:

```python
from psf import particle_filter

student_obs = data["observations"][0]
student_mask = data["masks"][0]

out = particle_filter(best_model, student_obs, student_mask, n_particles=1000)
filtered = out["filtered_probs"]   # (T, 6)
ess = out["ess"]                   # effective sample sizes

print("Filtered state probabilities (last 5 timesteps):")
for t in range(-5, 0):
    row = "  ".join(f"{v:.3f}" for v in filtered[t])
    print(f"  t={len(filtered)+t:3d}  {row}")
```

The particle filter is O(N) per timestep and scales to real-time deployment.

### Example 5: Running on Institutional Data

Replace the `load_data()` stub in `examples/02_real_data_template.py`:

```python
def load_data():
    """Load institutional data.

    Returns
    -------
    records : list[StudentRecord]
    observations : list[np.ndarray]
    masks : list[np.ndarray]
    """
    df = pd.read_parquet("data/au_courses.parquet")
    records, obs, masks = [], [], []
    for student_id, group in df.groupby("student_id"):
        records.append(build_student_record(group))
        o, m = prepare_student_observations(group)
        obs.append(o)
        masks.append(m)
    return records, obs, masks
```

Then run the standard pipeline.

---

## 10. The Four-Tier Validation Protocol

Empirical validation of the PSF requires four distinct tiers of evidence. Each tier answers a different question, and none substitutes for the others.

### Tier 1 — Synthetic Recovery (Implemented)

**Question:** Does the EM and inference machinery recover the true model when the data are generated by the model?

**Status:** Verified. Joint-state accuracy 0.859–0.912 with multi-restart, monotone EM convergence, correct parameter recovery after alignment, intent posterior AUC > 0.95.

**Limit:** This tier cannot detect model misspecification. If the real data are generated by a process that is not the PSF, synthetic recovery tells you nothing.

### Tier 2 — Operational Label Prediction (Implemented)

**Question:** Does the PSF predict the operational label better than baselines on held-out data?

**Design:**
- Temporal split (70/30)
- Leave-one-course-out
- Leave-one-cohort-out

**Metrics:**
- Primary: PR-AUC
- Secondary: matched-positive-rate precision, calibration (Brier, reliability diagram), recall of true returners

**Required ablation:** Fit without intent proxies (features 2–4) and with LMS-only (features 0, 5). If the ablation without intent proxies performs nearly as well as the full model, the intent construct is not empirically grounded.

### Tier 3 — Decision-Theoretic Evaluation (Proposed)

**Question:** Would decisions made using the PSF be better than decisions made using baselines?

**Design:** Retrospective decision analysis. For each student at a fixed decision point, compute the intervention each model would recommend, then evaluate:
- Intervention precision
- Missed returners
- Cost-weighted utility

This tier is where the PSF's distinctive claim lives: it should reduce unnecessary interventions on students who would have returned, without substantially increasing missed withdrawals.

### Tier 4 — Prospective Deployment (Future Work)

**Question:** Does using the PSF change student outcomes?

**Design:** Randomized or staggered deployment. Advisors see PSF outputs for a treatment cohort and baseline flags for a control cohort. Outcomes: return rate, time-to-return, completion rate, student satisfaction.

This is the only tier that can establish causal value. It requires institutional partnership and IRB approval.

### 10.1 The Key Pre-Modeling Analysis

Before any modeling, run this analysis to check whether intent proxies carry signal beyond inactivity:

```python
# Among students who stopped out, split by whether they had an intent proxy
stopped_out = (labels == 2) | (labels == 1)
has_extension = np.array([r.extension_active_after is not None for r in records])

return_rate_with = (labels[stopped_out & has_extension] == 2).mean()
return_rate_without = (labels[stopped_out & ~has_extension] == 2).mean()

print(f"Return rate with extension: {return_rate_with:.3f}")
print(f"Return rate without extension: {return_rate_without:.3f}")
```

If the difference is less than 10 percentage points, the intent proxies are not informative in this population, and the PSF will reduce to a three-state HMM with extra parameters. This is a **construct-validity check** that must be reported.

---

## 11. Interpreting Results

### 11.1 State Posteriors

For each student, the PSF produces a posterior distribution over the five persistence states at each time step. Interpretation:

- **High P(S2 = Progressing)**: Student is actively engaged.
- **High P(S3 = Stopped-Out)**: Student is inactive but may return.
- **High P(S3 with I=1)**: Student intends to return; light-touch intervention.
- **High P(S3 with I=0)**: Student likely to withdraw; escalate.
- **High P(S4 = Withdrawn)**: Student is effectively gone.

### 11.2 Intent-to-Return Posteriors

The `intent_posterior` function returns P(I=1 | history) at each timestep. Key patterns:

- **Rising intent**: Student is re-engaging; consider supportive check-in.
- **Falling intent**: Student is drifting toward withdrawal; consider proactive outreach.
- **Stable high intent**: Student is committed but paused; leave alone.

### 11.3 Return and Withdrawal Probabilities

The `return_probabilities` and `withdrawal_probabilities` functions produce horizon-specific probabilities. Use these for:

- **Early warning**: Flag students with P(Withdraw within 3 months) > 0.5.
- **Intervention triage**: Prioritize students with high withdrawal and low return probability.
- **Program evaluation**: Aggregate return probabilities to identify bottleneck courses.

### 11.4 Model Comparison

When comparing PSF to baselines, focus on:

- **PR-AUC**: Because withdrawal is rare, PR-AUC is more informative than AUC.
- **Matched-rate precision**: The operationally meaningful metric.
- **Recall of true returners**: The fraction of returners not flagged as at-risk. This protects against the most damaging failure mode.
- **Calibration**: For intervention decisions, probabilistic calibration matters more than ranking.

### 11.5 Sensitivity Analysis

Report results under multiple values of W and T_pause. If conclusions change materially, discuss the sensitivity in the paper.

### 11.6 What Counts as "Good" Performance

Rough benchmarks for self-paced institutions:

| Metric | Weak | Moderate | Strong | Excellent |
|---|---|---|---|---|
| PR-AUC (vs. baseline) | < 0.50 | 0.50–0.70 | 0.70–0.85 | > 0.85 |
| Matched-rate precision | < 0.60 | 0.60–0.75 | 0.75–0.85 | > 0.85 |
| Recall of true returners | < 0.50 | 0.50–0.65 | 0.65–0.80 | > 0.80 |
| Intent posterior AUC | < 0.60 | 0.60–0.75 | 0.75–0.90 | > 0.90 |

On the reference simulation with multi-restart, the PSF achieves PR-AUC 0.999, matched-rate precision 0.987, and intent AUC > 0.95. Real institutional data will almost certainly be lower; the goal is to substantially exceed the inactivity-threshold baseline.

---

## 12. Troubleshooting

### Problem 1: EM Does Not Converge

**Symptom**: Log-likelihood oscillates or decreases.

**Solutions**:
- Reduce the number of states or features.
- Increase `sigma_floor` to prevent variance collapse.
- Try multiple random initializations and pick the best.
- Check for degenerate sequences (e.g., all-missing students).

### Problem 2: All Students Assigned to One State

**Symptom**: Posterior is degenerate (all mass on one state).

**Solutions**:
- Check that emissions are distinguishable.
- Initialize emissions from data (e.g., using K-means).
- Increase the number of training students.

### Problem 3: Intent AUC Is Near 0.5

**Symptom**: Intent-to-Return is not recoverable from proxies.

**Solutions**:
- Verify that extension requests and tutor contacts are properly encoded.
- Check that these signals actually vary across students.
- Run the pre-modeling analysis (Section 10.1).
- Consider adding additional proxies.
- If no proxies are informative, report as a limitation.

### Problem 4: Survival Baseline Predicts Zero Positives

**Symptom**: `tp=0, fp=0` for `Survival (no intent)`.

**Cause**: Label-convention mismatch. The simulation uses (0=Completed, 1=Withdrawn, 2=Stopped-Out, 3=Censored) while `CompetingRiskSurvival` uses (0=censored, 1=return, 2=withdrawal, 3=completion).

**Solution**: Re-encode labels before passing them to the survival model:

```python
event_for_survival = np.zeros_like(labels)
event_for_survival[labels == 0] = 3
event_for_survival[labels == 1] = 2
event_for_survival[labels == 2] = 0
event_for_survival[labels == 3] = 0
```

### Problem 5: Multi-Restart Takes Too Long

**Symptom**: Five restarts × 30 iterations × 400 students is slow.

**Solutions**:
- Reduce `n_restarts` to 3 for exploration; use 5–10 for final reporting.
- Reduce `n_iter` to 20; the log-likelihood is usually near-plateau.
- Subsample training students.
- Use vectorized operations (already in the library).

### Problem 6: Figures Are Not Generated

**Symptom**: `AttributeError: module 'psf.viz' has no attribute 'plot_em_convergence'`.

**Solutions**:
- Confirm `psf/viz.py` exists and contains the plotting functions.
- Confirm `from . import viz` is in `psf/__init__.py`.
- Clear caches: `find . -name __pycache__ -exec rm -rf {} +`.
- Reinstall: `pip install -e ".[viz]"`.
- Verify: `python -c "import psf.viz; print([x for x in dir(psf.viz) if x.startswith('plot_')])"`.

### Problem 7: Overfitting

**Symptom**: Training log-likelihood much higher than test.

**Solutions**:
- Reduce the number of features.
- Add regularization to emissions (fix `log_sigma` floor).
- Use temporal cross-validation.
- Report test-set metrics only.

---

## 13. Ethics and Deployment

### 13.1 Transparency

Students should know:

- What data are collected.
- How persistence is measured.
- How the model is used.
- What interventions may follow.

### 13.2 Non-Punitiveness

PSF outputs should never be used to:

- Penalize students.
- Deny services.
- Restrict access.
- Rank students publicly.

They should be used to:

- Trigger supportive outreach.
- Allocate advising resources.
- Identify course-design bottlenecks.
- Inform institutional policy.

### 13.3 Bias Auditing

Before deployment, check for differential performance across:

- Age groups.
- Gender.
- Disability status.
- International vs. domestic students.
- Program type.

If the model performs worse for a subgroup, investigate and mitigate.

### 13.4 Data Security

- Store data in encrypted, access-controlled environments.
- De-identify before analysis whenever possible.
- Retain data only as long as necessary.
- Document all processing steps.

### 13.5 Student Consent

If possible, obtain informed consent for the use of persistence data in predictive modeling. Where consent is not feasible, ensure institutional review and student notification.

### 13.6 Human-in-the-Loop

The PSF should inform outreach, not replace it. Any decision with material consequences for a student should require advisor review.

### 13.7 Ongoing Monitoring

- **Shadow mode**: Run alongside the existing system without acting on outputs for one term.
- **Calibration monitoring**: Compute reliability diagrams monthly. If calibration error exceeds 0.05, refit.
- **Subgroup audits**: Check performance quarterly across demographic groups.

---

## 14. Extending the Library

### 14.1 Adding New Emission Families

The default emission is Gaussian. For count data (e.g., number of posts), use Poisson:

```python
from psf.model import PSFModel
import numpy as np
import scipy.special

class PoissonPSFModel(PSFModel):
    def log_emission_prob(self, observations, mask=None):
        rate = np.exp(self.mu)  # (K, D)
        log_b = np.zeros((observations.shape[0], self.n_states))
        for k in range(self.n_states):
            log_b[:, k] = np.sum(
                observations * np.log(rate[k]) - rate[k]
                - scipy.special.gammaln(observations + 1),
                axis=1,
            )
        return log_b
```

### 14.2 Adding Covariates (DBN)

For the DBN formulation, extend `PSFModel` with covariate-dependent transitions:

```python
class DBNPSFModel(PSFModel):
    def set_covariates(self, X_t):
        """X_t: (K, K, M) covariate features per transition."""
        self.X_t = X_t

    def _refresh_A(self):
        logits = np.einsum("ijm,m->ij", self.X_t, self.beta)
        logits = np.where(self.allowed > 0, logits, -np.inf)
        from psf.utils import softmax_rows
        self.A = softmax_rows(logits)
```

### 14.3 Adding Deep Extensions

Replace linear emissions with a neural network:

```python
import torch
import torch.nn as nn

class NeuralEmission(nn.Module):
    def __init__(self, n_states, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_states),
        )

    def forward(self, y):
        return torch.log_softmax(self.net(y), dim=-1)
```

Use this with a differentiable HMM implementation (e.g., Pyro, TensorFlow Probability) for end-to-end training.

### 14.4 Program-Level Extension

To model program-level persistence, add a hierarchical layer:

```python
class HierarchicalPSF:
    def __init__(self, n_courses):
        self.course_models = [PSFModel() for _ in range(n_courses)]
        self.program_model = PSFModel()

    def fit(self, course_data, program_data):
        for i, data in enumerate(course_data):
            baum_welch(self.course_models[i], *data)
        program_obs = self._aggregate(course_data)
        baum_welch(self.program_model, program_obs)
```

---

## 15. Frequently Asked Questions

**Q1: Do I need all six features?**

No. You can use any subset. The library adapts to whatever `n_features` you specify. However, Intent-to-Return inference works best with extension, tutor, and registration features.

**Q2: How many students do I need?**

At least 200 for stable EM. For reliable PR-AUC estimates, 500+ is better. For intent recovery with informative proxies, 1000+ is ideal.

**Q3: How long should the observation window be?**

At least 24 months for ground-truth labeling. For training, 12–36 months is typical.

**Q4: Can I use weekly data instead of monthly?**

Yes. The time grid is arbitrary. Weekly data gives finer resolution but requires more data.

**Q5: What if I don't have extension requests?**

The model still works, but Intent-to-Return will be inferred mostly from registration patterns. Report this as a limitation.

**Q6: How do I know if my model is any good?**

Compare PR-AUC to the inactivity-threshold baseline. If the PSF does not improve PR-AUC or reduce FPR, the intent proxies may not be informative in your population. Run the pre-modeling analysis in Section 10.1.

**Q7: Why does my single-restart fit perform worse than multi-restart?**

EM converges to local optima. On the reference simulation, single-restart AUC was 0.685; multi-restart AUC was 0.998. Always use at least 3–5 restarts and keep the best.

**Q8: Can I deploy this in production?**

The library is research-grade. For production, add logging, monitoring, and drift detection. Validate regularly. Use shadow mode before live deployment.

**Q9: What about GDPR / FERPA?**

Consult your institutional legal and privacy offices. De-identify data. Obtain IRB approval.

**Q10: How do I cite the library?**

Cite the companion manuscript and the repository (see the README for BibTeX).

**Q11: Why does the survival baseline predict zero positives?**

Label-convention mismatch. See Problem 4 in Section 12.

**Q12: Why does the inactivity threshold beat the PSF in the simulation?**

Because the simulation's label is a deterministic function of the terminal joint state, and that state is directly observable from the final value of the inactivity feature. The baseline is near-Bayes-optimal by construction. In real data, the label is genuinely latent. See Section 8.2.

**Q13: Can I contribute?**

Yes. Open an issue or pull request. Contributions to emissions, covariates, deep extensions, and real-world validation are welcome.

---

## 16. References

Athabasca University. (2003). *Completion rates for learner-paced undergraduate courses*. Athabasca University.

Bean, J. P., & Metzner, B. S. (1985). A conceptual model of nontraditional undergraduate student attrition. *Review of Educational Research*, 55(4), 485–540.

Kember, D. (1995). *Open learning courses for adults: A model of student progress*. Educational Technology Publications.

Murphy, K. P. (2002). *Dynamic Bayesian networks: Representation, inference and learning* [Doctoral dissertation, University of California, Berkeley].

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.

Simpson, O. (2013). Student retention in distance education: Are we failing our students? *Open Learning*, 28(2), 105–119.

Tinto, V. (1993). *Leaving college: Rethinking the causes and cures of student attrition* (2nd ed.). University of Chicago Press.

Wang, H. H. (2025). *Distinguishing stop-out from attrition in self-paced online learning: A computational persistence framework* [Manuscript].

---

## Appendix A: Quick Reference

### A.1 Minimal Working Example

```python
import numpy as np
from psf import (
    PSFModel, simulate_cohort, baum_welch,
    forward_backward, evaluate_predictions,
)
from psf.states import IDX

# Simulate
true_model = PSFModel(n_features=6)
data = simulate_cohort(true_model, n_students=400, max_time=36)

# Multi-restart fit
best_model, best_ll = None, -np.inf
for seed in range(5):
    rng = np.random.default_rng(seed)
    m = PSFModel(n_features=6)
    m.mu = rng.normal(size=m.mu.shape) * 0.5
    m.log_sigma = np.zeros_like(m.log_sigma)
    baum_welch(m, data["observations"], data["masks"], n_iter=30)
    ll = sum(forward_backward(m, o, mk)["log_lik"]
             for o, mk in zip(data["observations"], data["masks"]))
    if ll > best_ll:
        best_model, best_ll = m, ll

# Predict using terminal posterior
y = (data["final_label"] == 1).astype(int)
prob = np.array([
    forward_backward(best_model, o, m)["gamma"][-1, [IDX.S4, IDX.S3_NO_INTENT]].sum()
    for o, m in zip(data["observations"], data["masks"])
])

# Evaluate
print(evaluate_predictions(y, prob))
```

### A.2 State Reference

| Joint State | Index | Persistence State | Intent |
|---|---|---|---|
| S1_REGISTERED | 0 | Registered | — |
| S2_PROGRESSING | 1 | Progressing | — |
| S3_STOPPED_OUT_NO_INTENT | 2 | Stopped-Out | 0 |
| S3_STOPPED_OUT_INTENT | 3 | Stopped-Out | 1 |
| S4_WITHDRAWN | 4 | Withdrawn | — |
| S5_COMPLETED | 5 | Completed | — |

### A.3 Label Reference

| Label | Meaning |
|---|---|
| 0 | Completed |
| 1 | Withdrawn (operational attrition) |
| 2 | Stopped-Out (operational stop-out) |
| 3 | Censored / still enrolled |

### A.4 Default Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `W` | 24 months | Observation window for attrition |
| `T_pause` | 2 months | Minimum pause to be Stopped-Out |
| `sigma_floor` | 1e-3 | Minimum emission standard deviation |
| `n_iter` | 30 | Maximum EM iterations |
| `tol` | 1e-4 | EM convergence tolerance |
| `n_particles` | 1000 | Particle filter sample size |
| `ess_threshold` | 0.5 | Resampling threshold |
| `n_restarts` | 5 | Multi-restart count |

---

## Appendix B: Glossary

- **DBN**: Dynamic Bayesian Network.
- **EM**: Expectation-Maximization.
- **ESS**: Effective Sample Size.
- **HMM**: Hidden Markov Model.
- **Intent to Return (Iₜ)**: Latent variable capturing commitment to continue.
- **Matched-rate evaluation**: Comparing models at equal positive-class capacity, mirroring advisor workload constraints.
- **PR-AUC**: Precision-Recall Area Under Curve.
- **PSF**: Persistence-State Framework.
- **Stopped-Out**: Temporary interruption with intent to return.
- **Withdrawn**: Permanent exit with no intent to return.

---

*This user guide is a living document. For updates, see the repository at https://github.com/hongxueharriswang/psf_lib.*