# Repository Guidelines

## Project Structure & Module Organization

- `src/auto_annotator/`: core library (adapters, annotators, config, utils).
- `src/bbox_fixer/`, `src/osr_fixer/`: GUI/tools for reviewing/fixing annotations.
- `scripts/`: runnable utilities (batch processing, dataset maintenance, converters).
- `tests/`: `pytest` unit tests; `tests/manual_tests/` contains scripts meant to be run manually.
- `config/`: runtime configuration (`config.yaml`) and prompt templates (`config/prompts/`); local secrets live in `config/.env` (do not commit).
- `data/`: local datasets/outputs (treat as non-source artifacts unless explicitly requested).
- `docs/`, `examples/`: reference docs and sample metadata.

## Build, Test, and Development Commands

This repo uses `uv` with `pyproject.toml` (Python `>=3.12,<3.13`).

```bash
uv sync                          # install dependencies (and create/update venv)
uv run pytest                    # run the full test suite
uv run python scripts/batch_processing.py
uv run python scripts/bbox_fixer_cli.py  # launch BBoxFixer GUI
uv run ruff check .              # lint (if dev deps installed)
uv run black .                   # format (if dev deps installed)
```

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints preferred, no `Any` unless unavoidable.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, constants in `UPPER_SNAKE_CASE`.
- Formatting/linting: `black` + `ruff` (use defaults unless a repo config is added later).

## Testing Guidelines

- Framework: `pytest` (some tests may use `pytest-asyncio`).
- Naming: add unit tests under `tests/` as `test_*.py`; avoid relying on external APIs/credentials.
- Run locally with `uv run pytest` and ensure all tests pass before pushing.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits-style prefixes seen in history: `feat:`, `fix:`, `refactor:`, plus `Revert "..."` when reverting.
- PRs should include: purpose/impact, how to validate (commands + expected output), and any config changes (e.g., new keys in `config/config.yaml`).
- Never include secrets (`config/.env`) or large datasets/outputs in PRs; prefer scripts and small fixtures.
