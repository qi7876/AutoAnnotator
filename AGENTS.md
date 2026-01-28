# Repository Guidelines

## Project Structure & Module Organization

- `src/auto_annotator/`: core library (adapters, annotators, Gemini client, config, utils).
- `src/video_captioner/`: generates dense captions for `caption_data/` videos (see `docs/VIDEO_CAPTIONER.md`).
- `src/bbox_fixer/`, `src/osr_fixer/`: review/fix tools (GUI/CLI) for annotation artifacts.
- `scripts/`: runnable utilities (batch processing, dataset maintenance, caption generation).
- `config/`: runtime config (`config.yaml`) and prompt templates (`config/prompts/`, `config/caption_prompts/`).
- `tests/`: `pytest` unit tests; `tests/manual_tests/` for manual/credentialed runs.
- `docs/`, `examples/`: reference docs and sample metadata.

## Build, Test, and Development Commands

This repo uses `uv` with `pyproject.toml` (Python `>=3.12,<3.13`).

```bash
uv sync                         # install runtime deps
uv sync --extra dev             # install dev tooling (pytest/ruff/black)
uv run pytest                   # run full test suite
uv run pytest tests/test_config.py
uv run python scripts/batch_processing.py
uv run python scripts/generate_captions.py --config video_captioner_config.toml
uv run ruff check .             # lint (dev extra)
uv run black .                  # format (dev extra)
```

## Coding Style & Naming Conventions

- Python: 4-space indentation; prefer type hints and small, testable functions.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep user-facing strings/prompts in `config/` rather than hardcoding in code.

## Testing Guidelines

- Framework: `pytest` (some tests use `pytest-asyncio`).
- Tests should be deterministic and offline: avoid Gemini/GCS calls; use fakes/stubs and small local fixtures.
- Naming: `tests/test_*.py` for unit tests; keep integration-style scripts in `tests/manual_tests/`.

## Commit & Pull Request Guidelines

- Commit messages follow Conventional Commits-style prefixes used in history (`feat:`, `fix:`, `chore:`, `docs:`).
- PRs include: summary, validation steps (commands + expected output), and screenshots for GUI changes.
- Do not commit secrets (`config/.env`), caches (e.g., `.ruff_cache/`), or large datasets/outputs.
