#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


_NUMERIC_STEM = re.compile(r"^\d+$")


@dataclass(frozen=True)
class MoveStats:
    scanned_events: int = 0
    selected_files: int = 0
    moved: int = 0
    skipped_exists: int = 0
    missing: int = 0
    failed: int = 0


def _iter_event_dirs(dataset_root: Path):
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue
        for event_dir in sorted(sport_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            yield sport_dir, event_dir


def _is_event_root_asset(path: Path) -> bool:
    if not path.is_file():
        return False

    if path.name == "metainfo.json":
        return True

    if path.suffix.lower() not in (".mp4", ".json"):
        return False

    stem = path.stem
    return bool(_NUMERIC_STEM.match(stem))


def move_assets(
    *,
    dataset_root: Path,
    output_root: Path,
    apply: bool,
    overwrite: bool,
    sport: str | None,
    event: str | None,
    max_files: int | None,
) -> MoveStats:
    stats = MoveStats()

    def _with(**kwargs) -> MoveStats:
        return MoveStats(**{**stats.__dict__, **kwargs})

    moved_count = 0

    for sport_dir, event_dir in _iter_event_dirs(dataset_root):
        if sport is not None and sport_dir.name != sport:
            continue
        if event is not None and event_dir.name != event:
            continue

        stats = _with(scanned_events=stats.scanned_events + 1)

        assets = [p for p in sorted(event_dir.iterdir()) if _is_event_root_asset(p)]

        if not assets:
            stats = _with(missing=stats.missing + 1)
            continue

        for src in assets:
            dst = output_root / sport_dir.name / event_dir.name / src.name

            stats = _with(selected_files=stats.selected_files + 1)

            if dst.exists():
                if not overwrite:
                    stats = _with(skipped_exists=stats.skipped_exists + 1)
                    continue
                if apply:
                    try:
                        dst.unlink()
                    except OSError:
                        stats = _with(failed=stats.failed + 1)
                        continue

            if apply:
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    stats = _with(moved=stats.moved + 1)
                except Exception:
                    stats = _with(failed=stats.failed + 1)
            else:
                stats = _with(moved=stats.moved + 1)

            moved_count += 1
            if max_files is not None and moved_count >= max_files:
                return stats

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move event-root assets from data/Dataset/{sport}/{event}/ into an archive directory "
            "(numeric {n}.mp4/{n}.json and metainfo.json). Default is dry-run."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root (default: data/Dataset)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/original_dataset"),
        help="Archive root (default: data/original_dataset)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the move (default: dry-run)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files",
    )
    parser.add_argument(
        "--sport",
        default=None,
        help="Only process a specific sport directory name (optional)",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Only process a specific event directory name (optional)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Stop after moving N files (useful for testing)",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    output_root: Path = args.output_root

    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    if output_root.resolve().is_relative_to(dataset_root.resolve()):
        print(
            "output_root must not be inside dataset_root (refusing to move into Dataset tree)",
            file=sys.stderr,
        )
        return 2

    if args.max_files is not None and args.max_files < 0:
        print("--max-files must be >= 0", file=sys.stderr)
        return 2

    stats = move_assets(
        dataset_root=dataset_root,
        output_root=output_root,
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        sport=args.sport,
        event=args.event,
        max_files=args.max_files,
    )

    print("dataset_root :", dataset_root)
    print("output_root  :", output_root)
    print("apply        :", bool(args.apply))
    print("overwrite    :", bool(args.overwrite))
    if args.sport is not None:
        print("sport filter :", args.sport)
    if args.event is not None:
        print("event filter :", args.event)

    print("scanned_events:", stats.scanned_events)
    print("selected_files:", stats.selected_files)
    if args.apply:
        print("moved        :", stats.moved)
    else:
        print("would_move   :", stats.moved)
    print("skipped_exists:", stats.skipped_exists)
    print("missing_events:", stats.missing)
    print("failed       :", stats.failed)

    if stats.failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
