#!/usr/bin/env python3
"""Remove extracted temp frames generated from clip videos."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    dataset_root = Path("data/Dataset")
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        return 1

    temp_frames = sorted(dataset_root.glob("*/*/clips/*_frame_*.jpg"))
    if not temp_frames:
        print("No temp frames found.")
        return 0

    deleted = 0
    for frame_path in temp_frames:
        try:
            frame_path.unlink()
            deleted += 1
            print(f"deleted: {frame_path}")
        except OSError as exc:
            print(f"failed: {frame_path} ({exc})")

    print(f"deleted={deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
