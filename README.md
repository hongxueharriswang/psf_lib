# PSF: Persistence-State Framework

[![PyPI version](https://img.shields.io/badge/pypi-0.1.0-blue.svg)](https://pypi.org/project/psf-learning/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/hongxueharriswang/psf_lib/branch/main/graph/badge.svg)](https://codecov.io/gh/hongxueharriswang/psf)
[![Documentation](https://img.shields.io/badge/docs-your--org.github.io-blue)](https://hongxueharriswang.github.io/psf_lib/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A Python library for **distinguishing temporary stop-out from permanent attrition** in self-paced online learning.

The library implements the **Persistence-State Framework (PSF)** described in:

> Wang, [First Name]. (2025). *Distinguishing Stop-Out from Attrition in Self-Paced Online Learning: A Computational Persistence Framework* [Manuscript].

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Repository Structure](#repository-structure)
- [Core Concepts](#core-concepts)
- [Worked Example](#worked-example)
- [Command-Line Interface](#command-line-interface)
- [Performance Notes](#performance-notes)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Contributing](#contributing)
- [Ethics and Responsible Use](#ethics-and-responsible-use)
- [Acknowledgments](#acknowledgments)
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
| A | Registers, pauses for 3 months, returns, completes | Flagged as dropout at month 4 | High P(Return), low P(Withdraw) |
| B | Registers, becomes inactive, no extension, no contact | Flagged as dropout | High P(Withdraw) |
| C | Registers, formally withdraws | Correctly flagged | Correctly flagged |

The PSF distinguishes these cases; the threshold model does not. This matters operationally: a student in case A needs a gentle reminder, while a student in case B needs substantive re-engagement support.

---

## Key Features

- **Hidden Markov Model** over six joint states (five persistence states plus an intent split for S₃).
- **Baum-Welch (EM)** parameter estimation with structural constraints and missing-data handling.
- **Forward-backward**, **Viterbi**, and **particle filter** inference for offline and online use.
- **Return and withdrawal probabilities** over arbitrary horizons.
- **Competing-risk survival models** (discrete-time, cause-specific hazards).
- **Operational label generation** independent of the model, with configurable observation windows and pause thresholds.
- **Three baselines** for comparison: inactivity threshold, static logistic regression, survival without intent.
- **Evaluation metrics**: AUC, PR-AUC, Brier score, concordance index, time-dependent AUC, and matched-positive-rate evaluation.
- **Synthetic cohort simulation** for reproducible validation.
- **Type-hinted**, **tested**, and **documented** throughout.

---

## Installation

### From PyPI

```bash
pip install psf-learning
```

### From source

```bash
git clone https://github.com/hongxueharriswang/psf.git
cd psf
pip install -e .
```

### Development installation

```bash
git clone https://github.com/hongxueharriswang/psf.git
cd psf
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

### Requirements

- Python ≥ 3.9
- NumPy ≥ 1.22
- SciPy ≥ 1.9
- scikit-learn ≥ 1.1
- pandas ≥ 1.4
- matplotlib ≥ 3.5 (optional, for visualization)

---

## Quick Start

The fastest way to verify your installation is the simulation study:

```bash
python examples/01_simulation_study.py
```

This runs end-to-end: it builds a synthetic "true" model, simulates a cohort of students, re-fits the model via Baum-Welch, and compares the fitted model to three baselines.

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
data = simulate_cohort(true_model, n_students=300, max_time=36, missing_rate=0.2)

# 2. Fit the model via EM
model = baum_welch(
    PSFModel(n_features=6),
    data["observations"],
    data["masks"],
    n_iter=30,
    verbose=True,
)

# 3. Predict withdrawal probability
y = (data["final_label"] == 1).astype(int)
psf_prob = []
for seq, m in zip(data["observations"], data["masks"]):
    out = forward_backward(model, seq, m)
    g = out["gamma"]
    p_withdraw_t = g[:, IDX.S4] + g[:, IDX.S3_NO_INTENT]
    psf_prob.append(float(p_withdraw_t.max()))
psf_prob = np.asarray(psf_prob)

# 4. Evaluate
metrics = evaluate_predictions(y, psf_prob)
print(f"AUC: {metrics['auc']:.3f}   PR-AUC: {metrics['pr_auc']:.3f}")
```

---

## Documentation

Full documentation is available at [https://hongxueharriswang.github.io/psf_lib/](https://hongxueharriswang.github.io/psf_lib/).

| Document | Description |
|---|---|
| [User Guide](docs/user_guide.md) | Step-by-step guide to data preparation and validation. |
| [Theory](docs/theory.md) | Mathematical formulation of the PSF. |
| [API Reference](docs/api.md) | Complete class and function reference. |
| [Examples](docs/examples.md) | Runnable scripts with explanations. |

---

## Repository Structure

```
psf_lib/
├── psf_lib/                       # The library
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
│   ├── utils.py               # Numerical utilities
│   └── py.typed               # PEP 561 marker
├── examples/                  # Runnable scripts
│   ├── 01_simulation_study.py
│   ├── 02_real_data_template.py
│   ├── 03_intent_recovery.py
│   └── 04_online_tracking.py
├── tests/                     # Unit tests (pytest)
├── docs/                      # MkDocs documentation
├── .github/                   # CI workflows and issue templates
├── pyproject.toml             # Package configuration
├── requirements.txt
├── requirements-dev.txt
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── LICENSE
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

The latent variable Iₜ ∈ {0,1} captures the student's commitment to continue. It is inferred from proxies such as extension requests, tutor contacts, future course registrations, and communication with advising services. Intent is theoretically distinct from activity: a student may be inactive but still intend to return, or active but already intend to withdraw.

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

---

## Worked Example

A complete validation workflow, including data preparation, operational label generation, fitting, baseline comparison, and matched-rate evaluation, is provided in [`examples/01_simulation_study.py`](examples/01_simulation_study.py). The example is intentionally self-contained and reproducible with a fixed random seed.

A template for institutional data is in [`examples/02_real_data_template.py`](examples/02_real_data_template.py). Replace the `load_data()` stub with your own data loader, and the rest of the pipeline runs unchanged.

### Running on Institutional Data

To validate the PSF on your institution's records, you will need:

1. **Registration records** — student ID, course ID, registration date.
2. **Activity records** — student ID, timestamp, activity type.
3. **Assessment records** — submission times, scores.
4. **Intent proxies** — extension requests, tutor contacts, future registrations.
5. **Administrative records** — formal withdrawals, completions.

Then follow the seven steps in the [User Guide, Section 7](docs/user_guide.md#7-step-by-step-validation-workflow).

---

## Command-Line Interface

A convenience CLI is provided for common operations:

```bash
# Run a full simulation study
python -m psf.simulate --n-students 400 --max-time 36 --missing-rate 0.2

# Fit a model from a prepared dataset
python -m psf.fit --input data/observations.npz --output model.pkl

# Evaluate a fitted model
python -m psf.evaluate --model model.pkl --data data/test.npz
```

The CLI is a thin wrapper around the Python API; all functionality is accessible programmatically.

---

## Performance Notes

- **Complexity.** One EM iteration over N sequences of length T with K states and D features is O(N · T · K²) for the transition updates and O(N · T · K · D) for the emissions. With the default K = 6 and D = 6, the runtime is dominated by the forward-backward pass.
- **Recommended scale.** The library has been tested on cohorts of up to 10,000 students with sequences up to 60 timesteps. For larger datasets, consider vectorizing across students or using a distributed particle filter.
- **Convergence.** EM typically converges in 15–30 iterations on realistic data. Multiple restarts (5–10) are recommended to avoid local optima.
- **Missing data.** Missing features are marginalized out of the Gaussian emission; missing timesteps are propagated forward without a state update. Accuracy remains reasonable up to 40% missingness.
- **Numerical stability.** All recursions are computed in log space with the log-sum-exp trick. Variances are floored at 10⁻³ to prevent collapse.

---

## Roadmap

### v0.1.0 (current)

- [x] Hidden Markov Model over six joint states
- [x] Baum-Welch EM with structural constraints
- [x] Forward-backward, Viterbi, particle filter
- [x] Competing-risk survival models
- [x] Operational label generation
- [x] Baselines and evaluation metrics
- [x] Synthetic cohort simulation

### v0.2.0 (planned)

- [ ] Covariate-aware Dynamic Bayesian Network
- [ ] Additional emission families (Poisson, Bernoulli, ordinal probit)
- [ ] Integration with `lifelines` for continuous-time survival
- [ ] Interactive state-transition visualizations
- [ ] CLI for common operations

### v0.3.0 (planned)

- [ ] Deep extensions with PyTorch (recurrent emissions)
- [ ] Hierarchical program-level model
- [ ] Distributed particle filter for streaming data
- [ ] Published benchmark on AU data

### v1.0.0 (target)

- [ ] Stable API
- [ ] Comprehensive test coverage (> 90%)
- [ ] Peer-reviewed empirical validation paper
- [ ] Deployment guide for institutional use

---

## Citation

If you use this library in your research, please cite both the software and the companion manuscript.

**Software:**

```bibtex
@software{psf2025,
  title       = {PSF: A Python Library for the Persistence-State Framework},
  author      = {Wang, [First Name]},
  year        = {2025},
  version     = {0.1.0},
  doi         = {10.5281/zenodo.XXXXXXX},
  url         = {https://github.com/hongxueharriswang/psf}
}
```

**Manuscript:**

```bibtex
@article{wang2025persistence,
  title   = {Distinguishing Stop-Out from Attrition in Self-Paced Online Learning:
             A Computational Persistence Framework},
  author  = {Wang, [First Name]},
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

Areas where contributions are particularly welcome:

- Additional emission families (Poisson, Bernoulli, ordinal).
- Covariate-aware DBN implementations.
- Deep extensions with PyTorch or TensorFlow.
- Hierarchical program-level models.
- Visualization utilities.
- Real-world validation case studies.
- Documentation improvements.

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Ethics and Responsible Use

The PSF is designed for **research and institutional support**. Deployments should observe the following principles:

### Transparency

Students should know:

- What data are collected.
- How persistence is measured.
- How the model is used.
- What interventions may follow.

### Non-Punitiveness

PSF outputs should **never** be used to:

- Penalize students.
- Deny services.
- Restrict access.
- Rank students publicly.

They should be used to:

- Trigger supportive outreach.
- Allocate advising resources.
- Identify course-design bottlenecks.
- Inform institutional policy.

### Bias Auditing

Before deployment, check for differential performance across:

- Age groups.
- Gender.
- Disability status.
- International vs. domestic students.
- Program type.

If the model performs worse for a subgroup, investigate and mitigate.

### Data Security

- Store data in encrypted, access-controlled environments.
- De-identify before analysis whenever possible.
- Retain data only as long as necessary.
- Document all processing steps.

### Institutional Review

Obtain IRB (or equivalent) approval before accessing student records. See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## Acknowledgments

The framework was developed in the context of Athabasca University's learner-paced undergraduate programs. The author thanks colleagues in the Faculty of Science and Technology and the Centre for Learning Design and Development for discussions that shaped the operational definitions of stop-out and attrition.

The library builds on the following open-source projects:

- [NumPy](https://numpy.org/) — numerical computing
- [SciPy](https://scipy.org/) — scientific computing
- [scikit-learn](https://scikit-learn.org/) — baseline models and metrics
- [pandas](https://pandas.pydata.org/) — tabular data
- [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) — documentation

The implementation follows the formalisms in:

- Rabiner, L. R. (1989). A tutorial on hidden Markov models.
- Murphy, K. P. (2002). Dynamic Bayesian networks: Representation, inference and learning.
- Koller, D., & Friedman, N. (2009). Probabilistic graphical models.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.

```
MIT License

Copyright (c) 2025 [First Name] Wang

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

## Status

**Alpha.** The library is under active development. The API is not yet frozen. Breaking changes may occur before v1.0.0. Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/hongxueharriswang/psf_lib/issues).

---

## Contact

- **Issues:** [https://github.com/hongxueharriswang/psf_lib/issues](https://github.com/hongxueharriswang/psf_lib/issues)
- **Discussions:** [https://github.com/hongxueharriswang/psf_lib/discussions](https://github.com/hongxueharriswang/psf_lib/discussions)
- **Email:** harrisw@athabascau.ca

---

*Last updated: 2025-01-15*