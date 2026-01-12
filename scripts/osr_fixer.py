#!/usr/bin/env python3
from pathlib import Path
import sys

from osr_fixer.viewer import run_app


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "data" / "Dataset"
    output_root = project_root / "data" / "output"
    run_app(dataset_root, output_root)


if __name__ == "__main__":
    sys.exit(main())
