#!/usr/bin/env python3
"""Validate clip frame indices by checking timestamp monotonicity."""

from __future__ import annotations

import subprocess
from pathlib import Path


def check_monotonic_timestamps(video_path: Path, timeout_sec: int = 30) -> tuple[bool, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as exc:
        return False, f"ffprobe failed: {exc}"

    last_ts = None
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            ts = float(value)
        except ValueError:
            continue
        if last_ts is not None and ts < last_ts:
            return False, f"timestamp reversed: {last_ts} -> {ts}"
        last_ts = ts
    return True, "ok"


def main() -> int:
    dataset_root = Path("data/Dataset")
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        return 1

    clip_paths = sorted(dataset_root.glob("*/*/clips/*.mp4"))
    if not clip_paths:
        print("No clip videos found.")
        return 0

    failed = 0
    for clip_path in clip_paths:
        ok, message = check_monotonic_timestamps(clip_path)
        if not ok:
            failed += 1
            print(f"failed: {clip_path} ({message})")

    print(f"checked={len(clip_paths)} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
