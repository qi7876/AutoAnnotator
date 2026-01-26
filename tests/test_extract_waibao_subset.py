"""Tests for scripts/extract_waibao_subset.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "scripts" / "extract_waibao_subset.py"


def test_extract_subset_without_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    event_dir = dataset_root / "SportA" / "EventA"
    event_dir.mkdir(parents=True)

    (event_dir / "1.mp4").write_bytes(b"fake-mp4")
    (event_dir / "1.json").write_text('{"id": "1"}\n', encoding="utf-8")

    output_root = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--filename",
            "1.mp4",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    assert (output_root / "SportA" / "EventA" / "1.mp4").is_file()
    assert not (output_root / "SportA" / "EventA" / "1.json").exists()


def test_extract_subset_with_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    event_dir = dataset_root / "SportA" / "EventA"
    event_dir.mkdir(parents=True)

    (event_dir / "1.mp4").write_bytes(b"fake-mp4")
    (event_dir / "1.json").write_text('{"id": "1"}\n', encoding="utf-8")

    output_root = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--filename",
            "1.mp4",
            "--with-json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    exported_mp4 = output_root / "SportA" / "EventA" / "1.mp4"
    exported_json = output_root / "SportA" / "EventA" / "1.json"
    assert exported_mp4.is_file()
    assert exported_mp4.stat().st_size > 0
    assert exported_json.is_file()
    assert exported_json.stat().st_size > 0

