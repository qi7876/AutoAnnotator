#!/usr/bin/env python3
"""Launch BBoxFixer GUI."""

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from bbox_fixer.viewer import run_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="BBoxFixer GUI entrypoint")
    parser.add_argument(
        "--mode",
        choices=["normal", "flagged"],
        default="normal",
        help=(
            "normal: 原有模式; flagged: 仅加载 output 下标记 retrack/is_window_consistence 的任务"
        ),
    )
    args = parser.parse_args()

    dataset_root = repo_root / "data" / "Dataset"
    output_root = repo_root / "data" / "output"
    state_path = repo_root / "data" / "bbox_fixer_state.json"
    flagged_mode = args.mode == "flagged"
    run_app(dataset_root, output_root, state_path, flagged_mode=flagged_mode)


if __name__ == "__main__":
    main()
