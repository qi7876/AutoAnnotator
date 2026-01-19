#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_csv_set(value: str) -> set[str]:
    out: set[str] = set()
    for raw in value.split(","):
        name = raw.strip()
        if name:
            out.add(name)
    return out


def _iter_temp_frames(
    dataset_root: Path, kinds: set[str], exts: set[str]
) -> list[Path]:
    candidates: set[Path] = set()

    for kind in sorted(kinds):
        for ext in sorted(exts):
            candidates.update(dataset_root.glob(f"*/*/{kind}/*_frame_*.{ext}"))

    return sorted(p for p in candidates if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove extracted temp frames (*_frame_*.jpg) accidentally generated under clips/ or frames/ directories. "
            "Default is dry-run; use --apply to delete files."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root (default: data/Dataset)",
    )
    parser.add_argument(
        "--kinds",
        default="clips,frames",
        help="Comma-separated subdirectories to scan (default: clips,frames)",
    )
    parser.add_argument(
        "--exts",
        default="jpg,jpeg,png",
        help="Comma-separated extensions to delete (default: jpg,jpeg,png)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete files (default: dry-run)",
    )
    parser.add_argument(
        "--print",
        dest="print_paths",
        action="store_true",
        help="Print each matched path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N matches (useful for testing)",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"Dataset root not found or not a directory: {dataset_root}")
        return 2

    kinds = _parse_csv_set(args.kinds)
    exts = {e.lower().lstrip(".") for e in _parse_csv_set(args.exts)}

    if not kinds:
        print("--kinds is empty")
        return 2
    if not exts:
        print("--exts is empty")
        return 2

    temp_frames = _iter_temp_frames(dataset_root, kinds=kinds, exts=exts)
    if args.limit is not None:
        temp_frames = temp_frames[: max(0, args.limit)]

    print("dataset_root:", dataset_root)
    print("kinds       :", ",".join(sorted(kinds)))
    print("exts        :", ",".join(sorted(exts)))
    print("apply       :", bool(args.apply))
    print("matches     :", len(temp_frames))

    if not temp_frames:
        return 0

    deleted = 0
    failed = 0
    for frame_path in temp_frames:
        if args.print_paths:
            print(frame_path)

        if not args.apply:
            continue

        try:
            frame_path.unlink()
            deleted += 1
        except OSError as exc:
            failed += 1
            print(f"failed: {frame_path} ({exc})")

    if args.apply:
        print(f"deleted={deleted} failed={failed}")
        return 1 if failed else 0

    print("Run with --apply to delete these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
