#!/usr/bin/env python3
"""Launch MOT editor GUI."""

from pathlib import Path

from mot_editor.viewer import run_app


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / "data" / "Dataset"
    output_root = repo_root / "data" / "output" / "temp"
    state_path = repo_root / "data" / "mot_editor_state.json"
    run_app(dataset_root, output_root, state_path)


if __name__ == "__main__":
    main()
