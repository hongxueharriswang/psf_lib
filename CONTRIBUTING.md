# Contributing to PSF

Thank you for your interest in contributing to the Persistence-State Framework library.

## Code of Conduct

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Open an issue using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
Include:

- A minimal reproducible example.
- Your Python, NumPy, and scikit-learn versions.
- The full traceback.
- Expected vs. actual behavior.

### Suggesting Features

Open an issue using the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).
Describe:

- The use case.
- Why existing functionality is insufficient.
- Any proposed implementation.

### Submitting Pull Requests

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Install development dependencies: `pip install -r requirements-dev.txt`.
4. Install the pre-commit hooks: `pre-commit install`.
5. Make your changes.
6. Add tests in `tests/`.
7. Run the test suite: `pytest`.
8. Run linting: `ruff check psf tests`.
9. Run type checking: `mypy psf`.
10. Format code: `black psf tests`.
11. Commit with a descriptive message.
12. Push and open a pull request.

### Style Guidelines

- Follow PEP 8 and NumPy docstring conventions.
- Keep lines under 88 characters (Black default).
- Add type hints to all public functions.
- Write tests for all new functionality.
- Update the changelog under `[Unreleased]`.

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add new emission family`
- `fix: correct mask handling in forward pass`
- `docs: update user guide with examples`
- `test: add tests for particle filter`
- `refactor: simplify EM accumulator logic`

## Development Setup

```bash
git clone https://github.com/your-org/psf.git
cd psf
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
pytest