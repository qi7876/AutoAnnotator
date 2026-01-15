#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Change:
    path: Path
    removed: tuple[str, ...]


def _parse_csv_set(value: str) -> set[str]:
    items = [v.strip() for v in value.split(",")]
    return {v for v in items if v}


def _iter_segment_jsons(dataset_root: Path) -> Iterable[Path]:
    for kind in ("frames", "clips"):
        yield from sorted(dataset_root.glob(f"*/*/{kind}/*.json"))


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def cleanup_deprecated_tasks(
    *,
    segment_paths: list[Path],
    deprecated_tasks: set[str],
    apply: bool,
) -> tuple[list[Change], Counter[str], int]:
    changes: list[Change] = []
    removed_counts: Counter[str] = Counter()
    invalid_json = 0

    for json_path in segment_paths:
        data = _load_json(json_path)
        if not isinstance(data, dict):
            invalid_json += 1
            continue

        tasks_raw = data.get("tasks_to_annotate")
        if not isinstance(tasks_raw, list):
            continue

        removed_set: set[str] = set()
        kept_raw: list[Any] = []

        for item in tasks_raw:
            if isinstance(item, str):
                name = item.strip()
                if name in deprecated_tasks:
                    removed_set.add(name)
                    removed_counts[name] += 1
                    continue
            kept_raw.append(item)

        if not removed_set:
            continue

        data["tasks_to_annotate"] = kept_raw
        changes.append(Change(path=json_path, removed=tuple(sorted(removed_set))))

        if apply:
            _write_json(json_path, data)

    return changes, removed_counts, invalid_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove deprecated tasks from tasks_to_annotate in segment JSONs under data/Dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root directory (default: data/Dataset)",
    )
    parser.add_argument(
        "--deprecated-tasks",
        default="POS",
        help="Comma-separated deprecated task names to remove (default: POS)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes in-place (default: dry-run)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print changed files as JSON lines",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    deprecated_tasks = _parse_csv_set(args.deprecated_tasks)
    if not deprecated_tasks:
        print("--deprecated-tasks is empty", file=sys.stderr)
        return 2

    segment_paths = list(_iter_segment_jsons(dataset_root))
    changes, removed_counts, invalid_json = cleanup_deprecated_tasks(
        segment_paths=segment_paths,
        deprecated_tasks=deprecated_tasks,
        apply=bool(args.apply),
    )

    if args.json:
        for ch in changes:
            print(
                json.dumps(
                    {
                        "path": str(ch.path),
                        "removed": list(ch.removed),
                    },
                    ensure_ascii=False,
                )
            )
    else:
        print("dataset_root:", dataset_root)
        print("deprecated_tasks:", ",".join(sorted(deprecated_tasks)))
        print("apply:", bool(args.apply))
        print("scanned:", len(segment_paths))
        print("invalid_json:", invalid_json)
        print("changed_files:", len(changes))
        if removed_counts:
            for name, count in sorted(removed_counts.items()):
                print(f"removed_task\t{name}\t{count}")

        if changes and not args.apply:
            print("\nRun with --apply to write changes.")

    return 1 if changes else 0


if __name__ == "__main__":
    raise SystemExit(main())
