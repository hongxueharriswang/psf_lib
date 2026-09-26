# PSF: Persistence-State Framework


[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library for **distinguishing temporary stop-out from permanent attrition** in self-paced online learning.

The library implements the **Persistence-State Framework (PSF)** described in:

> Wang, H. H. (2025). *Distinguishing Stop-Out from Attrition in Self-Paced Online Learning: A Computational Persistence Framework* [Manuscript].

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Core Concepts](#core-concepts)
- [Example Results](#example-results)
- [Documentation](#documentation)
- [Validation Protocol](#validation-protocol)
- [Command Reference](#command-reference)
- [Citation](#citation)
- [Contributing](#contributing)
- [Ethics and Responsible Use](#ethics-and-responsible-use)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

In self-paced learning environments such as Athabasca University, students may pause their studies for weeks or months and later return. Traditional dropout models treat inactivity as withdrawal, which:

- **Overestimates attrition** by conflating temporary pause with permanent exit.
- **Misdirects interventions** by flagging students who intend to return.
- **Alienates adult learners** whose study patterns reflect work, family, and health commitments.

The PSF addresses this by:

1. Representing persistence as a **five-state dynamic process** — Registered (S₁), Progressing (S₂), Stopped-Out (S₃), Withdrawn (S₄), and Completed (S₅).
2. Introducing a latent **Intent-to-Return** variable (Iₜ) that distinguishes temporary stop-out from permanent attrition.
3. Providing **computational implementations** using Hidden Markov Models (HMMs), Dynamic Bayesian Networks (DBNs), survival analysis, and event history analysis.
4. Offering an **operational validation framework** with explicit ground-truth definitions, baselines, and evaluation metrics.

This library is the reference implementation for empirical validation of the framework. It is designed for researchers, institutional data scientists, and graduate students working in learning analytics and distance education.

---

## Why This Matters

In self-paced programs, the assumption "no activity = dropout" is demonstrably false. Consider three students observed over twelve months:

| Student | Trajectory | Inactivity threshold (30 days) | PSF |
|---|---|---|---|
| A | Registers, pauses 3 months, returns, completes | Flagged as dropout at month 4 | High P(Return), low P(Withdraw) |
| B | Registers, becomes inactive, no extension, no contact | Flagged as dropout | High P(Withdraw) |
| C | Registers, formally withdraws | Correctly flagged | Correctly flagged |

The PSF distinguishes these cases; the threshold model does not. This matters operationally: a student in case A needs a gentle reminder, while a student in case B needs substantive re-engagement support.

In a controlled simulation with 400 students, the PSF achieved **AUC 0.998** and **precision 0.987 at matched capacity**, versus AUC 0.881 and precision 0.895 for the best inactivity-threshold baseline. See [Example Results](#example-results) for details.

---

## Key Features

### Modeling

- **Hidden Markov Model** over six joint states (five persistence states plus an intent split for S₃).
- **Baum-Welch (EM)** parameter estimation with structural constraints and missing-data handling.
- **Multi-restart fitting** to avoid poor local optima; select the restart with the highest training log-likelihood.
- **Covariate support** via the Dynamic Bayesian Network extension.
- **Competing-risk survival models** (discrete-time, cause-specific hazards).

### Inference

- **Forward-backward** for offline state posteriors.
- **Viterbi** for most-likely state sequences.
- **Particle filter** for online state estimation and real-time tracking.
- **Return and withdrawal probabilities** over arbitrary horizons.
- **Intent-to-Return posterior** for diagnosing individual trajectories.

### Validation

- **Operational label generation** independent of the model, with configurable observation windows and pause thresholds.
- **Three baselines** for comparison: inactivity threshold, static logistic regression, survival without intent.
- **Evaluation metrics**: AUC, PR-AUC, Brier score, concordance index, time-dependent AUC, and matched-positive-rate evaluation.
- **Synthetic cohort simulation** for reproducible verification of the inference machinery.

### Diagnostics

- **Parameter recovery** with automatic state alignment (Hungarian algorithm).
- **EM convergence history** recorded on the fitted model.
- **Intent-proxy validity checks** for construct validation.
- **Calibration diagnostics** for deployment readiness.

### Visualization

- **Thirteen plotting functions** covering EM convergence, transition matrices, emission means, state trajectories, intent trajectories, ROC/PR curves, confusion matrices, matched-rate metrics, and cumulative withdrawal curves.
- Figures are publication-ready (150 DPI, vector-compatible) and saved to `./figures/`.

---

## Installation

### From PyPI

```bash
pip install psf-learning
```

### From source

```bash
git clone https://github.com/hongxueharriswang/psf_lib.git
cd psf_lib
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -e .
```

### With visualization support

```bash
pip install -e ".[viz]"
```

### With development dependencies

```bash
pip install -e ".[dev]"
pre-commit install
```

### Requirements

- **Core:** Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.9, scikit-learn ≥ 1.1, pandas ≥ 1.4
- **Visualization (optional):** matplotlib ≥ 3.5
- **Development:** pytest, black, ruff, mypy, pre-commit

---

## Quick Start

The fastest way to verify your installation is the simulation study:

```bash
python examples/01_simulation_study.py
```

This runs end-to-end: it builds a synthetic "true" model, simulates a cohort of 400 students, re-fits the model via Baum-Welch with five random restarts, evaluates parameter and state recovery, compares to three baselines, and produces thirteen diagnostic figures in `./figures/`.

Minimal Python usage:

```python
import numpy as np
from psf import (
    PSFModel, simulate_cohort, baum_welch,
    forward_backward, evaluate_predictions,
)
from psf.states import IDX

# 1. Simulate a cohort under the PSF
true_model = PSFModel(n_features=6)
data = simulate_cohort(true_model, n_students=400, max_time=36, missing_rate=0.2)

# 2. Fit the model via EM with multiple restarts
best_model, best_ll = None, -np.inf
for seed in range(5):
    rng = np.random.default_rng(seed)
    model = PSFModel(n_features=6)
    model.mu = rng.normal(size=model.mu.shape) * 0.5
    model.log_sigma = np.zeros_like(model.log_sigma)
    baum_welch(model, data["observations"], data["masks"], n_iter=30)
    ll = sum(forward_backward(model, o, m)["log_lik"]
             for o, m in zip(data["observations"], data["masks"]))
    if ll > best_ll:
        best_model, best_ll = model, ll

# 3. Predict withdrawal probability using the terminal posterior
y = (data["final_label"] == 1).astype(int)
psf_prob = []
for seq, m in zip(data["observations"], data["masks"]):
    out = forward_backward(best_model, seq, m)
    g = out["gamma"]
    psf_prob.append(float(g[-1, IDX.S4] + g[-1, IDX.S3_NO_INTENT]))
psf_prob = np.asarray(psf_prob)

# 4. Evaluate
metrics = evaluate_predictions(y, psf_prob)
print(f"AUC: {metrics['auc']:.3f}   PR-AUC: {metrics['pr_auc']:.3f}")
```

---

## Repository Layout

```
psf_lib/
├── psf/                       # The library
│   ├── __init__.py            # Public API
│   ├── states.py              # State definitions and structural constraints
│   ├── model.py               # PSFModel (HMM parameters)
│   ├── inference.py           # Forward-backward, Viterbi, particle filter
│   ├── learning.py            # Baum-Welch (EM) parameter estimation
│   ├── survival.py            # Competing-risk discrete-time survival
│   ├── labels.py              # Operational label generation
│   ├── baselines.py           # Baseline models for comparison
│   ├── evaluation.py          # Metrics and model comparison
│   ├── features.py            # Feature extraction
│   ├── simulation.py          # Synthetic cohort generation
│   ├── viz.py                 # Visualization utilities
│   ├── utils.py               # Numerical utilities
│   └── py.typed               # PEP 561 marker
├── examples/                  # Runnable scripts
│   ├── 01_simulation_study.py     # End-to-end verification
│   ├── 02_real_data_template.py   # Template for institutional data
│   ├── 03_intent_recovery.py      # Intent-to-Return validation
│   └── 04_online_tracking.py      # Particle-filter state tracking
├── tests/                     # Unit tests (pytest)
├── docs/                      # MkDocs documentation
│   ├── index.md
│   ├── user_guide.md
│   ├── theory.md
│   ├── api.md
│   └── examples.md
├── .github/workflows/         # CI configuration
├── pyproject.toml             # Package configuration
├── requirements.txt
├── requirements-dev.txt
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── LICENSE
└── README.md
```

---

## Core Concepts

### The Five Persistence States

| State | Label | Description |
|---|---|---|
| S₁ | Registered | Enrolled but not yet progressing. |
| S₂ | Progressing | Actively completing assessments. |
| S₃ | Stopped-Out | Temporarily inactive with some probability of returning. |
| S₄ | Withdrawn | Formally or effectively exited with no intent to return. |
| S₅ | Completed | Course requirements fulfilled. |

The critical distinction is between S₃ (transient) and S₄ (absorbing).

### Intent to Return

The latent variable Iₜ ∈ {0,1} captures the student's commitment to continue. It is inferred from proxies such as:

- Extension requests
- Tutor contacts
- Future course registrations
- Communication with advising services
- Submission of partially completed work

Intent is theoretically distinct from activity: a student may be inactive but still intend to return, or active but already intend to withdraw.

### The Joint State Space

The library represents six joint states, splitting S₃ by intent:

| Index | Joint State | Persistence | Intent |
|---|---|---|---|
| 0 | S1_REGISTERED | Registered | — |
| 1 | S2_PROGRESSING | Progressing | — |
| 2 | S3_STOPPED_OUT_NO_INTENT | Stopped-Out | 0 |
| 3 | S3_STOPPED_OUT_INTENT | Stopped-Out | 1 |
| 4 | S4_WITHDRAWN | Withdrawn | — |
| 5 | S5_COMPLETED | Completed | — |

### What the Model Computes

- **P(Return | History)** — probability the student returns to Progressing.
- **P(Withdraw | History)** — probability the student transitions to Withdrawn.
- **P(Iₜ = 1 | History)** — posterior probability of Intent to Return.
- **Time-varying state posteriors** — P(Zₜ = k | y₁:ₜ) for each joint state.

### Default Features

The `FeatureExtractor` produces six features per timestep:

| Index | Feature | Description |
|---|---|---|
| 0 | `activity` | `log1p(count of LMS activities)` |
| 1 | `submission` | 1 if assignment submitted this period |
| 2 | `extension` | 1 if extension requested |
| 3 | `tutor` | 1 if tutor contact occurred |
| 4 | `future_reg` | 1 if registered for a future course |
| 5 | `inactivity` | `log1p(time since last activity)` |

Features 2–4 are the intent proxies. Feature 5 is the inactivity signal used by the threshold baseline.

---

## Example Results

The simulation study (`examples/01_simulation_study.py`) produces results of the following form on a 400-student synthetic cohort. Numbers are from a representative run and will vary slightly with random seed.

### EM Convergence

Five random restarts, each converging in 25–30 iterations. The spread between the best and worst restart is approximately 980 log-likelihood units, confirming that multi-restart fitting is essential.

| Restart | Final log-likelihood |
|---|---|
| 0 | −41,352.95 |
| 1 | −40,860.32 |
| 2 | **−40,857.21** (selected) |
| 3 | −41,838.84 |
| 4 | −41,437.30 |

### Parameter Recovery

After state alignment (Hungarian algorithm), the fitted transition matrix closely matches the true one:

| From → To | True | Fitted |
|---|---|---|
| S1 → S2 | 0.88 | 0.91 |
| S2 → S2 | 0.82 | 0.85 |
| S3(I=0) → S3(I=0) | 0.79 | 0.81 |
| S3(I=1) → S2 | 0.35 | 0.33 |
| S3(I=1) → S3(I=1) | 0.57 | 0.58 |
| S4 → S4 | 1.00 | 1.00 |
| S5 → S5 | 1.00 | 1.00 |

Emission means recover cleanly for S1, S2, S3(I=0), S3(I=1), and S4. S5 shows a mild bias toward the population average, a small-sample effect from the absorbing state.

### Predictive Performance

At the default 0.5 threshold and at matched positive rate (59.8% of students, matching withdrawal prevalence):

| Model | AUC | PR-AUC | Precision (matched) |
|---|---|---|---|
| **PSF (posterior)** | **0.998** | **0.999** | **0.987** |
| PSF (Viterbi) | 0.975 | 0.980 | 0.950 |
| Inactivity threshold | 0.881 | 0.872 | 0.895 |
| Logistic regression | 0.913 | 0.937 | 0.862 |
| Survival (no intent) | 0.882 | 0.908 | 0.828 |

At the operational capacity of 239 students, the PSF flags 236 true withdrawals, compared to 214 for the inactivity threshold — 22 fewer unnecessary interventions.

### Intent Recovery

For a student with an oscillating trajectory (multiple stop-out and return cycles), the Intent-to-Return posterior tracks the true intent label with high accuracy. This is the mechanism that allows the framework to distinguish stop-out from attrition.

### Generated Figures

The simulation produces thirteen figures in `./figures/`:

| File | What it shows |
|---|---|
| `01_em_convergence.png` | Log-likelihood per iteration, one line per restart |
| `02_restart_scores.png` | Final LL per restart, best highlighted |
| `03_transition_matrices.png` | True vs fitted transition matrices |
| `04_emission_means.png` | Per-state true vs fitted emission means |
| `05_state_trajectory_student_*.png` | Posterior trajectories for four students |
| `06_intent_student_*.png` | Intent-to-Return posterior for one student |
| `07_roc_pr_curves.png` | ROC and PR curves for all five models |
| `08_confusion_matrices.png` | Confusion matrices at 0.5 threshold |
| `09_metric_bars.png` | AUC, PR-AUC, F1 bar chart |
| `10_matched_rate_metrics.png` | Precision, AUC, PR-AUC at matched rate |
| `11_withdrawal_curve.png` | Cumulative P(Withdrawn within h) for one student |

---

## Documentation

Full documentation is available at [https://hongxueharriswang.github.io/psf_lib/](https://hongxueharriswang.github.io/psf_lib/).

| Document | Description |
|---|---|
| [User Guide](docs/user_guide.md) | Step-by-step data preparation and validation |
| [Theory](docs/theory.md) | Mathematical formulation of the PSF |
| [API Reference](docs/api.md) | Complete class and function reference |
| [Examples](docs/examples.md) | Runnable scripts with explanations |

Key sections of the User Guide:

- **Section 4 — Data Collection**: required data streams and ethical considerations.
- **Section 5 — Data Preparation**: time grids, observation sequences, masks, operational labels.
- **Section 7 — Validation Workflow**: nine-step protocol from data preparation to sensitivity analysis.
- **Section 11 — Ethics and Deployment**: transparency, non-punitiveness, bias auditing.

---

## Validation Protocol

Empirical validation of the PSF on real institutional data follows a four-tier protocol. Tiers 1 and 2 are implemented in the library; Tiers 3 and 4 are proposed for institutional partnerships.

### Tier 1 — Synthetic Recovery (Implemented)

Verify that the EM and inference machinery recover the true model on synthetic data. Metrics: EM convergence, parameter recovery after alignment, joint-state accuracy, intent-posterior AUC.

### Tier 2 — Operational Label Prediction (Implemented)

Fit the PSF on held-out AU data and evaluate against three baselines using **temporal splits**, **leave-one-course-out**, and **leave-one-cohort-out** cross-validation.

**Primary metric:** PR-AUC, because withdrawal is rare.
**Secondary metrics:** matched-positive-rate precision, calibration (Brier score, reliability diagram), recall of true returners.

**Ablation study (required):** Fit the PSF three times, with the full feature set, without the intent proxies, and with LMS activity only. If the ablation without intent proxies performs nearly as well as the full model, the intent construct is not empirically grounded in the data set.

### Tier 3 — Decision-Theoretic Evaluation (Proposed)

Retrospective decision analysis. For each student at a fixed decision point, compute the intervention that each model would recommend, and evaluate:

- **Intervention precision** — fraction of interventions targeting true withdrawals.
- **Missed returners** — fraction of returners flagged for unnecessary intervention.
- **Cost-weighted utility** — expected utility per student given relative costs.

This tier is where the PSF's distinctive claim lives: it should reduce unnecessary interventions on students who would return, without substantially increasing missed withdrawals.

### Tier 4 — Prospective Deployment (Future Work)

Randomized or staggered deployment with institutional partners. Advisors see PSF outputs for a treatment cohort and baseline flags for a control cohort. Outcomes: return rate, time-to-return, completion rate, student satisfaction.

---

## Command Reference

### Running Examples

```bash
# End-to-end simulation study (recommended first run)
python examples/01_simulation_study.py

# Template for institutional data (edit load_data() first)
python examples/02_real_data_template.py

# Intent-to-Return recovery validation
python examples/03_intent_recovery.py

# Online state tracking with the particle filter
python examples/04_online_tracking.py
```

### Running Tests

```bash
pytest                              # Full test suite
pytest tests/test_learning.py -v    # Single module
pytest --cov=psf --cov-report=html  # With HTML coverage report
```

### Code Quality

```bash
ruff check psf tests examples       # Linting
black psf tests examples            # Formatting
mypy psf                            # Type checking
```

### Documentation

```bash
mkdocs serve                        # Local preview at http://localhost:8000
mkdocs build                        # Build static site
```

### Building and Publishing

```bash
python -m build                     # Build sdist and wheel
twine check dist/*                  # Validate before upload
twine upload dist/*                 # Publish to PyPI
```

---

## Citation

If you use this library in your research, please cite both the software and the companion manuscript.

**Software:**

```bibtex
@software{wang2025psf,
  title       = {PSF: A Python Library for the Persistence-State Framework},
  author      = {Wang, Hongxue Harris},
  year        = {2025},
  version     = {0.1.0},
  url         = {https://github.com/hongxueharriswang/psf_lib}
}
```

**Manuscript:**

```bibtex
@article{wang2025persistence,
  title   = {Distinguishing Stop-Out from Attrition in Self-Paced Online
             Learning: A Computational Persistence Framework},
  author  = {Wang, Hongxue Harris},
  journal = {[Journal Name]},
  year    = {2025},
  note    = {Manuscript}
}
```

See [`CITATION.cff`](CITATION.cff) for additional formats.

---

## Contributing

We welcome contributions from researchers, developers, and practitioners. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Reporting bugs
- Suggesting features
- Submitting pull requests
- Style and testing conventions

### Areas of Interest

- Additional emission families (Poisson, Bernoulli, ordinal probit)
- Covariate-aware Dynamic Bayesian Network
- Deep extensions with PyTorch (recurrent emissions)
- Hierarchical program-level models
- Distributed particle filter for streaming data
- Interactive state-transition visualizations
- Real-world validation case studies
- Documentation improvements

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Ethics and Responsible Use

The PSF is designed for **research and institutional support**. Deployments should observe the following principles.

### Transparency

Students should know:

- What data are collected
- How persistence is measured
- How the model is used
- What interventions may follow

### Non-Punitiveness

PSF outputs should **never** be used to:

- Penalize students
- Deny services
- Restrict access
- Rank students publicly

They should be used to:

- Trigger supportive outreach
- Allocate advising resources
- Identify course-design bottlenecks
- Inform institutional policy

### Bias Auditing

Before deployment, check for differential performance across age groups, gender, disability status, international vs. domestic students, and program type. If the model performs worse for a subgroup, investigate and mitigate.

### Data Security

- Store data in encrypted, access-controlled environments
- De-identify before analysis whenever possible
- Retain data only as long as necessary
- Document all processing steps

### Institutional Review

Obtain IRB (or equivalent) approval before accessing student records. See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## Roadmap

### v0.1.0 — Current Release

- [x] Hidden Markov Model over six joint states
- [x] Baum-Welch EM with structural constraints
- [x] Multi-restart fitting
- [x] Forward-backward, Viterbi, particle filter
- [x] Competing-risk survival models
- [x] Operational label generation
- [x] Baselines and evaluation metrics
- [x] Synthetic cohort simulation
- [x] Visualization module (thirteen plot functions)
- [x] Parameter-recovery diagnostics

### v0.2.0 — Planned

- [ ] Covariate-aware Dynamic Bayesian Network
- [ ] Additional emission families (Poisson, Bernoulli, ordinal)
- [ ] Integration with `lifelines` for continuous-time survival
- [ ] Interactive state-transition visualizations
- [ ] Command-line interface for common operations

### v0.3.0 — Planned

- [ ] Deep extensions with PyTorch (recurrent emissions)
- [ ] Hierarchical program-level model
- [ ] Distributed particle filter for streaming data
- [ ] Published benchmark on AU data

### v1.0.0 — Target

- [ ] Stable API
- [ ] Comprehensive test coverage (> 90%)
- [ ] Peer-reviewed empirical validation paper
- [ ] Deployment guide for institutional use

---

## Status

**Alpha.** The library is under active development. The API is not yet frozen. Breaking changes may occur before v1.0.0. Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/hongxueharriswang/psf_lib/issues).

### Known Limitations

1. **No empirical validation yet.** The library is verified on synthetic data; real-data validation is the next phase.
2. **Gaussian emissions only.** Poisson, Bernoulli, and ordinal emissions are planned for v0.2.0.
3. **Course-level only.** Program-level persistence is planned for v0.3.0.
4. **Python 3.9+ required.** Older versions are not supported.

---

## Contact

- **Issues:** [https://github.com/hongxueharriswang/psf_lib/issues](https://github.com/hongxueharriswang/psf_lib/issues)
- **Discussions:** [https://github.com/hongxueharriswang/psf_lib/discussions](https://github.com/hongxueharriswang/psf_lib/discussions)
- **Email:** hongxue.wang@example.com

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

```
MIT License

Copyright (c) 2025 Hongxue Harris Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*Last updated: 2025-01-15*