#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TASK_ABBR_TO_FULL: dict[str, str] = {
    "UCE": "Continuous_Events_Caption",
    "UCA": "Continuous_Actions_Caption",
    "USM": "ScoreboardMultiple",
    "STG": "Spatial_Temporal_Grounding",
    "USS": "ScoreboardSingle",
    "UOS": "Objects_Spatial_Relationships",
    "OSR": "Objects_Spatial_Relationships",
}
TASKS_TO_REMOVE: set[str] = {"RSI", "POT"}


@dataclass(frozen=True)
class Change:
    path: Path
    replaced: tuple[str, ...]
    removed: tuple[str, ...]


def _iter_jsons(dataset_root: Path) -> Iterable[Path]:
    yield from sorted(dataset_root.rglob("*.json"))


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_task_list(
    raw_tasks: list[Any],
) -> tuple[list[Any], tuple[str, ...], tuple[str, ...], bool]:
    replaced: set[str] = set()
    removed: set[str] = set()
    out: list[Any] = []
    seen: set[str] = set()
    changed = False

    for item in raw_tasks:
        if not isinstance(item, str):
            out.append(item)
            continue

        name = item.strip()
        if not name:
            out.append(item)
            continue

        if name in TASKS_TO_REMOVE:
            removed.add(name)
            changed = True
            continue

        mapped = TASK_ABBR_TO_FULL.get(name, name)
        if mapped != name:
            replaced.add(name)
            changed = True

        if mapped in seen:
            changed = True
            continue
        seen.add(mapped)
        out.append(mapped)

    return out, tuple(sorted(replaced)), tuple(sorted(removed)), changed


def _update_task_field(data: dict[str, Any], key: str) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], bool]:
    raw = data.get(key)
    if not isinstance(raw, list):
        return data, (), (), False

    normalized, replaced, removed, changed = _normalize_task_list(raw)
    if not changed:
        return data, (), (), False

    out = dict(data)
    out[key] = normalized
    return out, replaced, removed, True


def normalize_task_abbreviations(
    *,
    json_paths: list[Path],
    apply: bool,
) -> tuple[list[Change], Counter[str], Counter[str], int]:
    changes: list[Change] = []
    replaced_counter: Counter[str] = Counter()
    removed_counter: Counter[str] = Counter()
    invalid_json = 0

    for json_path in json_paths:
        data = _load_json(json_path)
        if data is None:
            invalid_json += 1
            continue

        changed_any = False
        replaced_all: set[str] = set()
        removed_all: set[str] = set()
        out = data

        for key in ("tasks_to_annotate", "task_to_annotate"):
            out, replaced, removed, changed = _update_task_field(out, key)
            if not changed:
                continue
            changed_any = True
            replaced_all.update(replaced)
            removed_all.update(removed)

        if not changed_any:
            continue

        for abbr in replaced_all:
            replaced_counter[abbr] += 1
        for abbr in removed_all:
            removed_counter[abbr] += 1

        if apply:
            _write_json(json_path, out)

        changes.append(
            Change(
                path=json_path,
                replaced=tuple(sorted(replaced_all)),
                removed=tuple(sorted(removed_all)),
            )
        )

    return changes, replaced_counter, removed_counter, invalid_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Expand abbreviated tasks in dataset JSONs: "
            "UCE/UCA/USM/STG/USS/UOS -> full task names; remove RSI/POT."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root directory (default: data/Dataset)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in-place (default: dry-run)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print changed files as JSON lines",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"dataset_root not found or not a directory: {dataset_root}", file=sys.stderr)
        return 2

    json_paths = list(_iter_jsons(dataset_root))
    changes, replaced_counter, removed_counter, invalid_json = normalize_task_abbreviations(
        json_paths=json_paths,
        apply=bool(args.apply),
    )

    if args.json:
        for ch in changes:
            print(
                json.dumps(
                    {
                        "path": str(ch.path),
                        "replaced_abbreviations": list(ch.replaced),
                        "removed_abbreviations": list(ch.removed),
                    },
                    ensure_ascii=False,
                )
            )
    else:
        print("dataset_root:", dataset_root)
        print("apply:", bool(args.apply))
        print("scanned:", len(json_paths))
        print("invalid_json:", invalid_json)
        print("changed_files:", len(changes))
        for abbr, count in sorted(replaced_counter.items()):
            print(f"replaced_abbreviation\t{abbr}\t{count}")
        for abbr, count in sorted(removed_counter.items()):
            print(f"removed_abbreviation\t{abbr}\t{count}")

        if changes and not args.apply:
            print("\nRun with --apply to write changes.")

    return 1 if changes else 0


if __name__ == "__main__":
    raise SystemExit(main())
