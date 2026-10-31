# Contributing

Thanks for your interest in improving this project.

## Development setup

```bash
git clone https://github.com/minalotfy0-web/health-classifier-diabetes-prediction-pytorch.git
cd health-classifier-diabetes-prediction-pytorch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -e ".[dev]"
```

Python 3.11 or newer is required.

## Checks

CI runs the same checks on Python 3.11, 3.12 and 3.13:

```bash
ruff check .
ruff format --check .
pytest
```

Optionally install the pre-commit hooks so ruff runs on every commit:

```bash
pre-commit install
```

## Pull requests

1. Open an issue first for larger changes.
2. Branch from `main` and keep each PR focused.
3. Add or update tests for behaviour changes.
4. Open a pull request against `main` and make sure CI is green.

Please do not change committed results or config numbers without explaining how they were regenerated.

By contributing you agree that your contributions are licensed under the project's MIT license. Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
