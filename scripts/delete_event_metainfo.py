#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stats:
    scanned_events: int = 0
    found: int = 0
    deleted: int = 0
    failed: int = 0


def _iter_events(dataset_root: Path):
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue
        for event_dir in sorted(sport_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            yield sport_dir, event_dir


def delete_metainfo(
    *,
    dataset_root: Path,
    apply: bool,
    sport: str | None,
    event: str | None,
    max_events: int | None,
) -> Stats:
    stats = Stats()

    def _with(**kwargs) -> Stats:
        return Stats(**{**stats.__dict__, **kwargs})

    done = 0

    for sport_dir, event_dir in _iter_events(dataset_root):
        if sport is not None and sport_dir.name != sport:
            continue
        if event is not None and event_dir.name != event:
            continue

        stats = _with(scanned_events=stats.scanned_events + 1)

        path = event_dir / "metainfo.json"
        if not path.exists():
            done += 1
            if max_events is not None and done >= max_events:
                break
            continue

        if not path.is_file():
            stats = _with(failed=stats.failed + 1)
            done += 1
            if max_events is not None and done >= max_events:
                break
            continue

        stats = _with(found=stats.found + 1)

        if apply:
            try:
                path.unlink()
                stats = _with(deleted=stats.deleted + 1)
            except OSError:
                stats = _with(failed=stats.failed + 1)
        else:
            stats = _with(deleted=stats.deleted + 1)

        done += 1
        if max_events is not None and done >= max_events:
            break

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete event-root metainfo.json under data/Dataset/{sport}/{event}/. Dry-run unless --apply."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root (default: data/Dataset)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete files (default: dry-run)",
    )
    parser.add_argument(
        "--sport",
        default=None,
        help="Only process a specific sport directory name",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Only process a specific event directory name",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after N events (debug)",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    stats = delete_metainfo(
        dataset_root=dataset_root,
        apply=bool(args.apply),
        sport=args.sport,
        event=args.event,
        max_events=args.max_events,
    )

    print("dataset_root:", dataset_root)
    print("apply       :", bool(args.apply))
    if args.sport is not None:
        print("sport filter:", args.sport)
    if args.event is not None:
        print("event filter:", args.event)

    print("scanned_events:", stats.scanned_events)
    print("found       :", stats.found)
    if args.apply:
        print("deleted     :", stats.deleted)
    else:
        print("would_delete:", stats.deleted)
    print("failed      :", stats.failed)

    return 0 if stats.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
