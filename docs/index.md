# PSF: Persistence-State Framework

Welcome to the documentation for the **PSF** Python library.

## What is the PSF?

The Persistence-State Framework (PSF) is a computational framework for
distinguishing **temporary stop-out** from **permanent attrition** in
self-paced online learning.

## Why Does It Matter?

In self-paced programs, students may pause for weeks or months and later
return. Traditional dropout models treat inactivity as withdrawal, which
overestimates attrition and misdirects interventions.

## What Does the Library Provide?

- A Hidden Markov Model over five persistence states.
- A latent Intent-to-Return variable.
- Baum-Welch (EM) parameter estimation.
- Forward-backward, Viterbi, and particle filter inference.
- Competing-risk survival models.
- Operational label generation.
- Baseline models for comparison.
- Evaluation metrics (AUC, PR-AUC, Brier, concordance).
- Synthetic data generation.

## Getting Started

See the [User Guide](user_guide.md) for step-by-step instructions.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{psf2025,
  title = {PSF: A Python Library for the Persistence-State Framework},
  author = {Wang, [First Name]},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/your-org/psf}
}