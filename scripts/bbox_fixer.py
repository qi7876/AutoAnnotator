#!/usr/bin/env python3
"""Launch BBoxFixer GUI."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from bbox_fixer.viewer import run_app  # noqa: E402


def main() -> None:
    dataset_root = repo_root / "data" / "Dataset"
    output_root = repo_root / "data" / "output"
    state_path = repo_root / "data" / "bbox_fixer_state.json"
    run_app(dataset_root, output_root, state_path)


if __name__ == "__main__":
    main()
