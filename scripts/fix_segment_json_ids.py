#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Stats:
    scanned: int = 0
    changed: int = 0
    skipped: int = 0
    invalid_json: int = 0


def _iter_segment_jsons(dataset_root: Path, kinds: set[str]) -> Iterable[Path]:
    for kind in sorted(kinds):
        yield from sorted(dataset_root.glob(f"*/*/{kind}/*.json"))


def _parse_csv_set(value: str) -> set[str]:
    out: set[str] = set()
    for raw in value.split(","):
        name = raw.strip()
        if name:
            out.add(name)
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def _normalize_id(value: Any) -> str | None:
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def fix_ids(
    *,
    dataset_root: Path,
    kinds: set[str],
    apply: bool,
    sport: str | None,
    event: str | None,
    limit: int | None,
) -> tuple[Stats, list[Path]]:
    stats = Stats()
    changed_paths: list[Path] = []

    for p in _iter_segment_jsons(dataset_root, kinds=kinds):
        if limit is not None and stats.scanned >= limit:
            break

        try:
            rel = p.relative_to(dataset_root)
        except ValueError:
            stats = Stats(
                scanned=stats.scanned + 1,
                changed=stats.changed,
                skipped=stats.skipped,
                invalid_json=stats.invalid_json,
            )
            continue

        if len(rel.parts) < 4:
            stats = Stats(
                scanned=stats.scanned + 1,
                changed=stats.changed,
                skipped=stats.skipped,
                invalid_json=stats.invalid_json,
            )
            continue

        sport_name, event_name = rel.parts[0], rel.parts[1]
        if sport is not None and sport_name != sport:
            continue
        if event is not None and event_name != event:
            continue

        stats = Stats(
            scanned=stats.scanned + 1,
            changed=stats.changed,
            skipped=stats.skipped,
            invalid_json=stats.invalid_json,
        )

        try:
            data = _load_json(p)
        except Exception:
            stats = Stats(
                scanned=stats.scanned,
                changed=stats.changed,
                skipped=stats.skipped,
                invalid_json=stats.invalid_json + 1,
            )
            continue

        if not isinstance(data, dict):
            stats = Stats(
                scanned=stats.scanned,
                changed=stats.changed,
                skipped=stats.skipped,
                invalid_json=stats.invalid_json + 1,
            )
            continue

        expected_id = p.stem
        current_id = _normalize_id(data.get("id"))
        if current_id == expected_id:
            stats = Stats(
                scanned=stats.scanned,
                changed=stats.changed,
                skipped=stats.skipped + 1,
                invalid_json=stats.invalid_json,
            )
            continue

        data["id"] = expected_id
        changed_paths.append(p)
        stats = Stats(
            scanned=stats.scanned,
            changed=stats.changed + 1,
            skipped=stats.skipped,
            invalid_json=stats.invalid_json,
        )

        if apply:
            _write_json(p, data)

    return stats, changed_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fix segment metadata JSON ids so that data['id'] matches the filename stem (e.g. clips/1.json -> id='1')."
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
        default="clips",
        help="Comma-separated segment kinds to fix (default: clips; options: clips,frames)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in-place (default: dry-run)",
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
        "--limit",
        type=int,
        default=None,
        help="Only scan first N JSON files (debug)",
    )
    parser.add_argument(
        "--print",
        dest="print_paths",
        action="store_true",
        help="Print each file that would be changed",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    kinds = _parse_csv_set(args.kinds)
    if not kinds:
        print("--kinds is empty", file=sys.stderr)
        return 2

    unknown = kinds - {"clips", "frames"}
    if unknown:
        print("Unknown kinds: " + ",".join(sorted(unknown)), file=sys.stderr)
        return 2

    stats, changed_paths = fix_ids(
        dataset_root=dataset_root,
        kinds=kinds,
        apply=bool(args.apply),
        sport=args.sport,
        event=args.event,
        limit=args.limit,
    )

    print("dataset_root:", dataset_root)
    print("kinds       :", ",".join(sorted(kinds)))
    print("apply       :", bool(args.apply))
    if args.sport is not None:
        print("sport filter:", args.sport)
    if args.event is not None:
        print("event filter:", args.event)

    print("scanned     :", stats.scanned)
    print("invalid_json:", stats.invalid_json)
    print("unchanged   :", stats.skipped)
    if args.apply:
        print("changed     :", stats.changed)
    else:
        print("would_change:", stats.changed)

    if args.print_paths:
        for p in changed_paths:
            print(p)

    return 0 if stats.invalid_json == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
