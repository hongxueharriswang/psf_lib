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
8. [Worked Examples](#8-worked-examples)
9. [Interpreting Results](#9-interpreting-results)
10. [Troubleshooting](#10-troubleshooting)
11. [Ethics and Deployment](#11-ethics-and-deployment)
12. [Extending the Library](#12-extending-the-library)
13. [Frequently Asked Questions](#13-frequently-asked-questions)
14. [References](#14-references)

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

The PSF does **not** provide:

- Pre-trained models (these must be fitted on institutional data).
- A dashboard or user interface.
- Automated intervention delivery.

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
| \(S_1\) | Registered | Enrolled but not yet progressing. |
| \(S_2\) | Progressing | Actively completing assessments. |
| \(S_3\) | Stopped-Out | Temporarily inactive with probability of return. |
| \(S_4\) | Withdrawn | Effectively exited with no intent to return. |
| \(S_5\) | Completed | Course requirements fulfilled. |

The critical distinction is between \(S_3\) (temporary) and \(S_4\) (permanent).

### 2.3 Intent to Return

The latent variable \(I_t \in \{0,1\}\) captures the student's commitment to continue. It is inferred from proxies such as:

- Extension requests.
- Tutor contacts.
- Future course registrations.
- Communication with advising services.

Intent to Return is theoretically distinct from activity. A student may be inactive but still intend to return.

### 2.4 What the Model Computes

The PSF computes:

- \( \Pr(\text{Return} \mid \text{History}) \): probability the student returns to Progressing.
- \( \Pr(\text{Withdraw} \mid \text{History}) \): probability the student withdraws.
- \( \Pr(I_t = 1 \mid \text{History}) \): posterior probability the student intends to return.
- Time-varying state posteriors \( \Pr(Z_t = k \mid y_{1:t}) \).

These outputs support early warning, intervention triage, and program evaluation.

### 2.5 Computational Formalisms

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
- matplotlib ≥ 3.5 (for visualization)

### 3.2 Installation

**Option A: From source (recommended for development)**

```bash
git clone https://github.com/your-org/psf.git
cd psf
pip install -e .
```

**Option B: Direct install**

```bash
pip install numpy scipy scikit-learn pandas matplotlib
# then copy the psf/ package into your project
```

### 3.3 Verify Installation

```python
import psf
print(psf.__version__)
# Output: 0.1.0

from psf import PSFModel, simulate_cohort, baum_welch
print("PSF library loaded successfully.")
```

### 3.4 Package Structure

```
psf/
├── states.py        # State definitions and structural constraints
├── model.py         # PSFModel (HMM parameters)
├── inference.py     # Forward-backward, Viterbi, particle filter
├── learning.py      # Baum-Welch (EM) parameter estimation
├── survival.py      # Competing-risk discrete-time survival
├── labels.py        # Operational label generation
├── baselines.py     # Baseline models for comparison
├── evaluation.py    # AUC, PR-AUC, Brier, concordance
├── features.py      # Feature extraction
└── simulation.py    # Synthetic data generation
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
| **Intent proxies** | Extension requests, tutor contacts, future registrations | Infers \(I_t\) |
| **Administrative** | Formal withdrawals, completion records | Defines S4 and S5 |

### 4.2 Minimum Viable Dataset

At a minimum, you need:

- **Registration records**: student ID, course ID, registration date.
- **Activity records**: student ID, timestamp, activity type.
- **Withdrawal records**: student ID, withdrawal date (if applicable).
- **Completion records**: student ID, completion date (if applicable).

With only these four, you can fit a basic PSF. Adding extension requests and tutor contacts significantly improves Intent-to-Return inference.

### 4.3 Data Collection Checklist

Before proceeding, confirm:

- [ ] Data are de-identified and ethically approved.
- [ ] Each student has a unique identifier.
- [ ] Timestamps are in a consistent timezone and format.
- [ ] Course start and end dates are recorded.
- [ ] Withdrawal and completion records are complete.
- [ ] Extension requests are logged.
- [ ] Tutor contacts are logged (if available).
- [ ] Observation window is at least 24 months for ground-truth labeling.

### 4.4 Ethical Considerations

Per Section 9.2 of the manuscript:

- Students should know how persistence is measured.
- Interventions should support rather than surveil.
- Data should be stored securely and de-identified.
- Model outputs should not be used punitively.

Obtain institutional review board (IRB) approval before accessing student records.

---

## 5. Data Preparation

### 5.1 Time Grid

The PSF operates on a discrete time grid. Choose:

- **Weekly grid** for early warning applications.
- **Monthly grid** for program evaluation.

The time grid defines the resolution of state transitions. All timestamps must be mapped to grid indices.

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
        time_since = t - student_df[student_df.event_type.isin(
            ["login", "submission"])].t.max() if len(student_df) > 0 else 0

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

### 5.7 Data Splitting

Use **temporal splitting** to avoid data leakage:

```python
# Sort by registration time
sorted_idx = np.argsort([r.registration_times[0] for r in records])
train_idx = sorted_idx[:int(0.7 * len(sorted_idx))]
test_idx = sorted_idx[int(0.7 * len(sorted_idx)):]

train_obs = [obs[i] for i in train_idx]
train_masks = [masks[i] for i in train_idx]
test_obs = [obs[i] for i in test_idx]
test_masks = [masks[i] for i in test_idx]
```

Do **not** split randomly. Self-paced programs have temporal drift in course design, student populations, and institutional policies.

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
│  → AUC, PR-AUC, Brier, FPR, concordance                     │
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
| `SurvivalWithoutIntent` | Baseline: survival without \(I_t\) |

### 6.3 Key Functions

| Function | Purpose |
|---|---|
| `baum_welch` | Fit PSFModel via EM |
| `forward_backward` | Compute state posteriors |
| `viterbi` | Most likely state sequence |
| `particle_filter` | Online state estimation |
| `return_probabilities` | P(Return) over horizon |
| `withdrawal_probabilities` | P(Withdraw) over horizon |
| `generate_operational_labels` | Create ground-truth labels |
| `evaluate_predictions` | Compute metrics |
| `compare_models` | Build comparison table |
| `simulate_cohort` | Generate synthetic data |

---

## 7. Step-by-Step Validation Workflow

### Step 1: Prepare Data

```python
import numpy as np
from psf import StudentRecord, FeatureExtractor

# Load institutional data
records = load_student_records()          # list of StudentRecord
obs, masks = load_observation_sequences() # lists of (T, D) arrays
```

### Step 2: Generate Operational Labels

```python
from psf import generate_operational_labels

labels = generate_operational_labels(
    records, W=24, T_pause=2, observation_end=36,
)

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

### Step 4: Initialize and Fit the PSF

```python
from psf import PSFModel, baum_welch

model = PSFModel(n_features=obs[0].shape[1])

# Optional: initialize emissions from data
model.mu = np.array([obs[i].mean(axis=0) for i in train_idx[:6]])

# Fit via EM
baum_welch(
    model,
    [obs[i] for i in train_idx],
    [masks[i] for i in train_idx],
    n_iter=50,
    verbose=True,
)
```

### Step 5: Inspect the Fitted Model

```python
# Transition matrix
print("Transition matrix (rows = from-state):")
import pandas as pd
from psf.states import JOINT_STATE_LABELS

df = pd.DataFrame(
    model.A,
    index=JOINT_STATE_LABELS,
    columns=JOINT_STATE_LABELS,
)
print(df.round(3))
```

Expected output:

```
                              S1: Registered  S2: Progressing  S3(I=0)  S3(I=1)  S4: Withdrawn  S5: Completed
S1: Registered                      0.750            0.250     0.000    0.000          0.000         0.000
S2: Progressing                     0.000            0.850     0.050    0.050          0.020         0.030
S3: Stopped-Out (I=0)               0.000            0.000     0.700    0.100          0.200         0.000
S3: Stopped-Out (I=1)               0.000            0.400     0.000    0.500          0.100         0.000
S4: Withdrawn                       0.000            0.000     0.000    0.000          1.000         0.000
S5: Completed                       0.000            0.000     0.000    0.000          0.000         1.000
```

### Step 6: Compute State and Intent Posteriors

```python
from psf import forward_backward, collapse_posterior, intent_posterior

# Example student
student_obs = obs[test_idx[0]]
student_mask = masks[test_idx[0]]

out = forward_backward(model, student_obs, student_mask)
gamma = out["gamma"]                    # (T, 6) joint-state posterior
persistence = collapse_posterior(gamma) # (T, 5)
intent = intent_posterior(gamma)        # (T,)

print("Persistence posteriors (last 5 timesteps):")
print(pd.DataFrame(
    persistence[-5:],
    columns=["Registered", "Progressing", "Stopped-Out", "Withdrawn", "Completed"],
).round(3))
```

### Step 7: Compute Return and Withdrawal Probabilities

```python
from psf import return_probabilities, withdrawal_probabilities

p_return = return_probabilities(model, gamma, horizon=6)
p_withdraw = withdrawal_probabilities(model, gamma, horizon=6)

print("P(Return within 6 months):", p_return[-1, -1].round(3))
print("P(Withdraw within 6 months):", p_withdraw[-1, -1].round(3))
```

### Step 8: Evaluate Against Baselines

```python
from psf import (
    InactivityThresholdClassifier,
    LogisticDropoutModel,
    SurvivalWithoutIntent,
    evaluate_predictions,
    compare_models,
)

y_test = (labels[test_idx] == 1).astype(int)  # binary: withdrawn vs not

# PSF-based withdrawal probability
psf_prob = []
for i in test_idx:
    out = forward_backward(model, obs[i], masks[i])
    g = out["gamma"][-1]
    psf_prob.append(g[3] + g[2])  # S3(I=1) + S3(I=0)
psf_prob = np.asarray(psf_prob)

# Features for baselines
X_test = np.array([obs[i].mean(axis=0) for i in test_idx])
inactivity = X_test[:, 5]

# Baselines
results = {}
results["PSF (fitted)"] = evaluate_predictions(y_test, psf_prob)
results["Inactivity threshold"] = evaluate_predictions(
    y_test,
    InactivityThresholdClassifier(threshold=1.5).predict_proba(inactivity)[:, 1],
)
results["Logistic regression"] = evaluate_predictions(
    y_test,
    LogisticDropoutModel().fit(X_test, y_test).predict_proba(X_test)[:, 1],
)
results["Survival (no intent)"] = evaluate_predictions(
    y_test,
    SurvivalWithoutIntent().fit(
        X_test,
        np.array([r.last_activity_time for r in [records[i] for i in test_idx]]),
        y_test,
    ).predict_cumulative_incidence(X_test, horizon=12)[:, -1, 1],
)

table = compare_models(results)
print(table.to_string(float_format=lambda x: f"{x:.3f}"))
```

### Step 9: Sensitivity Analysis

```python
for W in [12, 24, 36]:
    for T_pause in [1, 2, 3]:
        labels = generate_operational_labels(
            records, W=W, T_pause=T_pause, observation_end=36,
        )
        # ... re-run steps 3-8
        print(f"W={W}, T_pause={T_pause}: "
              f"Withdrawn={(labels == 1).sum()}, "
              f"Stopped-Out={(labels == 2).sum()}")
```

---

## 8. Worked Examples

### Example 1: Simulation Study

The simplest way to validate the library is via synthetic data.

```python
import numpy as np
from psf import PSFModel, simulate_cohort, baum_welch, forward_backward
from psf.states import IDX

# Build a "true" model
true_model = PSFModel(n_features=6)

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
true_model.A_logits = A_logits
true_model._refresh_A()

rng = np.random.default_rng(42)
true_model.mu = rng.normal(size=(6, 6))
true_model.mu[IDX.S2] += np.array([1.0, 0.8, 0.0, 0.0, 0.0, -0.5])
true_model.mu[IDX.S3_NO_INTENT] += np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 1.0])
true_model.mu[IDX.S3_INTENT] += np.array([-0.5, -0.3, 0.5, 0.5, 0.8, 0.8])
true_model.log_sigma = np.log(np.full((6, 6), 0.7))

# Simulate cohort
data = simulate_cohort(true_model, n_students=400, max_time=36, missing_rate=0.2)

# Fit model
init = PSFModel(n_features=6)
fitted = baum_welch(init, data["observations"], data["masks"], n_iter=30, verbose=True)

# Evaluate state recovery
correct, total = 0, 0
for seq, m, true_states in zip(data["observations"], data["masks"], data["states"]):
    out = forward_backward(fitted, seq, m)
    pred = out["gamma"].argmax(axis=1)
    correct += int((pred == true_states).sum())
    total += len(true_states)
print(f"State recovery accuracy: {correct / total:.3f}")
```

### Example 2: Validating Intent-to-Return Recovery

A key test of the PSF is whether the latent Intent-to-Return variable can be recovered from proxies.

```python
from psf import intent_posterior
from sklearn.metrics import roc_auc_score

# Collect posterior intent probabilities for all students
all_intent, all_true_intent = [], []
for seq, m, true_states in zip(
    data["observations"], data["masks"], data["states"]
):
    out = forward_backward(fitted, seq, m)
    intent = intent_posterior(out["gamma"])
    # True intent: 1 if state is S3_INTENT
    true_intent = (true_states == IDX.S3_INTENT).astype(float)
    # Only evaluate at S3 timesteps
    s3_mask = (true_states == IDX.S3_NO_INTENT) | (true_states == IDX.S3_INTENT)
    all_intent.extend(intent[s3_mask])
    all_true_intent.extend(true_intent[s3_mask])

all_intent = np.nan_to_num(all_intent)
auc = roc_auc_score(all_true_intent, all_intent)
print(f"Intent-to-Return AUC: {auc:.3f}")
```

Expected: AUC > 0.80 if proxies are informative.

### Example 3: Comparing to Baselines (Full Evaluation)

```python
from psf import (
    generate_operational_labels, evaluate_predictions, compare_models,
    InactivityThresholdClassifier, LogisticDropoutModel, SurvivalWithoutIntent,
)

labels = generate_operational_labels(records, W=24, T_pause=2, observation_end=36)
y = (labels == 1).astype(int)

# PSF probability
psf_prob = np.array([
    forward_backward(model, obs[i], masks[i])["gamma"][-1, [2, 3]].sum()
    for i in range(len(obs))
])

# Baselines
X = np.array([o.mean(axis=0) for o in obs])
results = {
    "PSF": evaluate_predictions(y, psf_prob),
    "Inactivity": evaluate_predictions(y, (X[:, 5] >= 1.5).astype(float)),
    "Logistic": evaluate_predictions(y, LogisticDropoutModel().fit(X, y).predict_proba(X)[:, 1]),
}

table = compare_models(results)
print(table[["auc", "pr_auc", "brier", "fpr", "tpr"]].round(3))
```

Expected pattern:

| Model | AUC | PR-AUC | Brier | FPR | TPR |
|---|---|---|---|---|---|
| PSF | 0.89 | 0.76 | 0.11 | 0.08 | 0.82 |
| Inactivity | 0.70 | 0.41 | 0.22 | 0.35 | 0.65 |
| Logistic | 0.84 | 0.67 | 0.16 | 0.18 | 0.75 |

The PSF should show **higher PR-AUC** and **lower FPR** than baselines.

### Example 4: Handling Missing Data

The PSF handles missingness via masks. Here is how to simulate and recover from 30% missing data:

```python
# Simulate with high missingness
data = simulate_cohort(true_model, n_students=400, missing_rate=0.3)

# Fit
fitted = baum_welch(PSFModel(n_features=6), data["observations"], data["masks"], n_iter=30)

# Evaluate
correct, total = 0, 0
for seq, m, true_states in zip(data["observations"], data["masks"], data["states"]):
    out = forward_backward(fitted, seq, m)
    pred = out["gamma"].argmax(axis=1)
    correct += int((pred == true_states).sum())
    total += len(true_states)
print(f"Accuracy with 30% missing: {correct / total:.3f}")
```

The forward-backward algorithm automatically marginalizes missing features, so accuracy should remain reasonable even at 30–40% missingness.

### Example 5: Online Tracking with Particle Filter

For real-time applications, use the particle filter:

```python
from psf import particle_filter

student_obs = obs[test_idx[0]]
student_mask = masks[test_idx[0]]

out = particle_filter(model, student_obs, student_mask, n_particles=1000)
filtered = out["filtered_probs"]   # (T, 6)
ess = out["ess"]

print("Filtered state probabilities (last 5 timesteps):")
print(pd.DataFrame(
    filtered[-5:],
    columns=JOINT_STATE_LABELS,
).round(3))
```

The particle filter is O(N) per timestep and scales to real-time deployment.

---

## 9. Interpreting Results

### 9.1 State Posteriors

For each student, the PSF produces a posterior distribution over the five persistence states at each time step. Interpretation:

- **High P(S2 = Progressing)**: Student is actively engaged.
- **High P(S3 = Stopped-Out)**: Student is inactive but may return.
- **High P(S3 with I=1)**: Student intends to return; light-touch intervention.
- **High P(S3 with I=0)**: Student likely to withdraw; escalate.
- **High P(S4 = Withdrawn)**: Student is effectively gone.

### 9.2 Intent-to-Return Posteriors

The `intent_posterior` function returns P(I=1 | history) at each timestep. Key patterns:

- **Rising intent**: Student is re-engaging; consider supportive check-in.
- **Falling intent**: Student is drifting toward withdrawal; consider proactive outreach.
- **Stable high intent**: Student is committed but paused; leave alone.

### 9.3 Return and Withdrawal Probabilities

The `return_probabilities` and `withdrawal_probabilities` functions produce horizon-specific probabilities. Use these for:

- **Early warning**: Flag students with \( \Pr(\text{Withdraw within 3 months}) > 0.5 \).
- **Intervention triage**: Prioritize students with high withdrawal and low return probability.
- **Program evaluation**: Aggregate return probabilities to identify bottleneck courses.

### 9.4 Model Comparison

When comparing PSF to baselines, focus on:

- **PR-AUC**: Because withdrawal is rare, PR-AUC is more informative than AUC.
- **FPR**: The PSF should substantially reduce false positives.
- **Brier score**: Calibration matters for intervention decisions.
- **Concordance index**: For survival models, C-index measures ranking quality.

### 9.5 Sensitivity Analysis

Report results under multiple values of \(W\) and \(T_{\text{pause}}\). If conclusions change materially, discuss the sensitivity in the paper.

---

## 10. Troubleshooting

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

- Check that emissions are distinguishable. If `mu` is identical across states, the model cannot differentiate.
- Initialize emissions from data (e.g., using K-means).
- Increase the number of training students.

### Problem 3: Intent AUC Is Near 0.5

**Symptom**: Intent-to-Return is not recoverable from proxies.

**Solutions**:

- Verify that extension requests and tutor contacts are properly encoded.
- Check that these signals actually vary across students.
- Consider adding additional proxies (e.g., advising contacts, registration inquiries).
- Report this as a limitation.

### Problem 4: Masking Is Incorrect

**Symptom**: Model treats missing data as observed zeros.

**Solutions**:

- Ensure `mask[t, d] = False` for missing features.
- For entirely missing timesteps, set all features to `False`.
- Do not set missing values to zero; the mask handles it.

### Problem 5: Slow EM Convergence

**Symptom**: EM takes a long time.

**Solutions**:

- Reduce `n_iter` and use early stopping (`tol=1e-3`).
- Subsample training students.
- Use vectorized NumPy operations (already done in library).
- Consider variational inference for large datasets.

### Problem 6: Overfitting

**Symptom**: Training log-likelihood much higher than test.

**Solutions**:

- Reduce the number of features.
- Add regularization to emissions (fix `log_sigma` floor).
- Use temporal cross-validation.
- Report test-set metrics only.

---

## 11. Ethics and Deployment

### 11.1 Transparency

Students should know:

- What data are collected.
- How persistence is measured.
- How the model is used.
- What interventions may follow.

### 11.2 Non-Punitiveness

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

### 11.3 Bias Auditing

Before deployment, check for differential performance across:

- Age groups.
- Gender.
- Disability status.
- International vs. domestic students.
- Program type.

If the model performs worse for a subgroup, investigate and mitigate.

### 11.4 Data Security

- Store data in encrypted, access-controlled environments.
- De-identify before analysis whenever possible.
- Retain data only as long as necessary.
- Document all processing steps.

### 11.5 Student Consent

If possible, obtain informed consent for the use of persistence data in predictive modeling. Where consent is not feasible, ensure institutional review and student notification.

---

## 12. Extending the Library

### 12.1 Adding New Emission Families

The default emission is Gaussian. For count data (e.g., number of posts), use Poisson:

```python
from psf.model import PSFModel
import numpy as np

class PoissonPSFModel(PSFModel):
    def log_emission_prob(self, observations, mask=None):
        # observations: (T, D) integer counts
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

### 12.2 Adding Covariates (DBN)

For the DBN formulation, extend `PSFModel` with covariate-dependent transitions:

```python
class DBNPSFModel(PSFModel):
    def set_covariates(self, X_t):
        """X_t: (K, K, M) covariate features per transition."""
        self.X_t = X_t

    def _refresh_A(self):
        # Compute logits as a linear function of covariates
        logits = np.einsum("ijm,m->ij", self.X_t, self.beta)
        logits = np.where(self.allowed > 0, logits, -np.inf)
        self.A = softmax_rows(logits)
```

### 12.3 Adding Deep Extensions

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

### 12.4 Program-Level Extension

To model program-level persistence, add a hierarchical layer:

```python
class HierarchicalPSF:
    def __init__(self, n_courses):
        self.course_models = [PSFModel() for _ in range(n_courses)]
        self.program_model = PSFModel()

    def fit(self, course_data, program_data):
        # Fit course-level models
        for i, data in enumerate(course_data):
            baum_welch(self.course_models[i], *data)
        # Aggregate course posteriors as observations for program model
        program_obs = self._aggregate(course_data)
        baum_welch(self.program_model, program_obs)
```

---

## 13. Frequently Asked Questions

**Q1: Do I need all six features?**

No. You can use any subset. The library adapts to whatever `n_features` you specify. However, Intent-to-Return inference works best with extension, tutor, and registration features.

**Q2: How many students do I need?**

At least 200 for stable EM. For reliable PR-AUC estimates, 500+ is better.

**Q3: How long should the observation window be?**

At least 24 months for ground-truth labeling. For training, 12–36 months is typical.

**Q4: Can I use weekly data instead of monthly?**

Yes. The time grid is arbitrary. Weekly data gives finer resolution but requires more data.

**Q5: What if I don't have extension requests?**

The model still works, but Intent-to-Return will be inferred mostly from registration patterns. Report this as a limitation.

**Q6: How do I know if my model is any good?**

Compare PR-AUC to the inactivity-threshold baseline. If the PSF does not improve PR-AUC or reduce FPR, the proxies may not be informative.

**Q7: Can I deploy this in production?**

The library is research-grade. For production, add logging, monitoring, and drift detection. Validate regularly.

**Q8: What about GDPR / FERPA?**

Consult your institutional legal and privacy offices. De-identify data. Obtain IRB approval.

**Q9: How do I cite the library?**

Cite the companion manuscript and the repository:

```bibtex
@software{psf2025,
  title = {PSF: A Python Library for the Persistence-State Framework},
  author = {Wang, [First Name]},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/your-org/psf}
}
```

**Q10: Can I contribute?**

Yes. Open an issue or pull request. Contributions to emissions, covariates, and deep extensions are welcome.

---

## 14. References

Athabasca University. (2003). *Completion rates for learner-paced undergraduate courses*. Athabasca University.

Bean, J. P., & Metzner, B. S. (1985). A conceptual model of nontraditional undergraduate student attrition. *Review of Educational Research*, 55(4), 485–540.

Kember, D. (1995). *Open learning courses for adults: A model of student progress*. Educational Technology Publications.

Murphy, K. P. (2002). *Dynamic Bayesian networks: Representation, inference and learning* [Doctoral dissertation, University of California, Berkeley].

Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.

Simpson, O. (2013). Student retention in distance education: Are we failing our students? *Open Learning*, 28(2), 105–119.

Tinto, V. (1993). *Leaving college: Rethinking the causes and cures of student attrition* (2nd ed.). University of Chicago Press.

Wang, [First Name]. (2025). *Distinguishing stop-out from attrition in self-paced online learning: A computational persistence framework* [Manuscript].

---

## Appendix A: Quick Reference

### A.1 Minimal Working Example

```python
import numpy as np
from psf import PSFModel, simulate_cohort, baum_welch, forward_backward, evaluate_predictions

# 1. Simulate
true_model = PSFModel(n_features=6)
data = simulate_cohort(true_model, n_students=300)

# 2. Fit
model = baum_welch(PSFModel(n_features=6), data["observations"], data["masks"], n_iter=20)

# 3. Predict
y = (data["final_label"] == 1).astype(int)
prob = np.array([
    forward_backward(model, o, m)["gamma"][-1, [2, 3]].sum()
    for o, m in zip(data["observations"], data["masks"])
])

# 4. Evaluate
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
| `n_iter` | 50 | Maximum EM iterations |
| `tol` | 1e-4 | EM convergence tolerance |
| `n_particles` | 1000 | Particle filter sample size |
| `ess_threshold` | 0.5 | Resampling threshold |

---

## Appendix B: Glossary

- **DBN**: Dynamic Bayesian Network.
- **EM**: Expectation-Maximization.
- **ESS**: Effective Sample Size.
- **HMM**: Hidden Markov Model.
- **Intent to Return (\(I_t\))**: Latent variable capturing commitment to continue.
- **PR-AUC**: Precision-Recall Area Under Curve.
- **PSF**: Persistence-State Framework.
- **Stopped-Out**: Temporary interruption with intent to return.
- **Withdrawn**: Permanent exit with no intent to return.

---

*This user guide is a living document. For updates, see the repository at https://github.com/your-org/psf.*