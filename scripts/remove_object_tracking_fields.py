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
    removed_tasks_to_annotate: tuple[str, ...]
    added_tasks_to_annotate: tuple[str, ...]
    removed_task_fields: tuple[str, ...]


def _parse_csv_set(value: str) -> set[str]:
    items = [v.strip() for v in value.split(",")]
    return {v for v in items if v}


def _parse_csv_list(value: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        name = raw.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _iter_default_jsons(dataset_root: Path) -> Iterable[Path]:
    yield from sorted(dataset_root.glob("*/*/frames/*.json"))
    yield from sorted(dataset_root.glob("*/*/clips/*.json"))


def _iter_glob(repo_root: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(repo_root.glob(pattern))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _update_tasks_to_annotate(
    obj: Any,
    from_tasks: set[str],
    replace_with: list[str],
    remove_only: bool,
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(obj, dict):
        return obj, (), ()

    raw = obj.get("tasks_to_annotate")
    if not isinstance(raw, list):
        return obj, (), ()

    needs_change = any(
        isinstance(item, str) and item.strip() in from_tasks for item in raw
    )
    if not needs_change:
        return obj, (), ()

    removed: set[str] = set()
    added: set[str] = set()
    out: list[Any] = []
    seen: set[str] = set()

    for item in raw:
        if isinstance(item, str):
            name = item.strip()
            if name in from_tasks:
                removed.add(name)
                if not remove_only:
                    for rep in replace_with:
                        rep_name = rep.strip()
                        if rep_name and rep_name not in seen:
                            seen.add(rep_name)
                            out.append(rep_name)
                            added.add(rep_name)
                continue

            if name and name not in seen:
                seen.add(name)
                out.append(item)
            continue

        out.append(item)

    new_obj = dict(obj)
    new_obj["tasks_to_annotate"] = out
    return new_obj, tuple(sorted(removed)), tuple(sorted(added))


def _is_task_dict(d: dict[str, Any], tasks: set[str]) -> bool:
    for key in ("task", "task_name", "taskName"):
        v = d.get(key)
        if isinstance(v, str) and v.strip() in tasks:
            return True
    return False


def _clean_object(obj: Any, tasks: set[str], removed_fields: set[str]) -> Any | None:
    if isinstance(obj, dict):
        if _is_task_dict(obj, tasks):
            return None

        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k in tasks:
                removed_fields.add(k)
                continue
            cleaned = _clean_object(v, tasks, removed_fields)
            if cleaned is None:
                continue
            out[k] = cleaned
        return out

    if isinstance(obj, list):
        out_list: list[Any] = []
        for item in obj:
            if isinstance(item, str) and item.strip() in tasks:
                removed_fields.add(item.strip())
                continue
            cleaned = _clean_object(item, tasks, removed_fields)
            if cleaned is None:
                continue
            out_list.append(cleaned)
        return out_list

    return obj


def remove_task_fields_in_json(
    *,
    data: Any,
    tasks: set[str],
    remove_keys_and_task_items: bool,
) -> tuple[Any, tuple[str, ...]]:
    removed_fields: set[str] = set()

    if remove_keys_and_task_items:
        cleaned = _clean_object(data, tasks, removed_fields)
        if cleaned is None:
            cleaned = data
    else:
        cleaned = data

    return cleaned, tuple(sorted(removed_fields))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace/remove tasks in dataset JSONs: update tasks_to_annotate and optionally drop task-specific fields."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root directory (default: data/Dataset)",
    )
    parser.add_argument(
        "--tasks",
        default="Object_Tracking",
        help="Comma-separated task names to replace/remove (default: Object_Tracking)",
    )
    parser.add_argument(
        "--replace-with",
        default="Spatial_Temporal_Grounding,Continuous_Actions_Caption",
        help=(
            "Comma-separated tasks to insert when a --tasks entry is found "
            "(default: Spatial_Temporal_Grounding,Continuous_Actions_Caption)"
        ),
    )
    parser.add_argument(
        "--remove-only",
        action="store_true",
        help="Remove --tasks from tasks_to_annotate without adding replacements",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra repo-root-relative glob patterns to include (repeatable)",
    )
    parser.add_argument(
        "--remove-task-fields",
        action="store_true",
        help=(
            "Also remove dict keys equal to task name, list items equal to task name, "
            "and dict elements with task/task_name/taskName matching the removed tasks"
        ),
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
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    from_tasks = _parse_csv_set(args.tasks)
    if not from_tasks:
        print("--tasks is empty", file=sys.stderr)
        return 2

    replace_with = _parse_csv_list(args.replace_with)
    remove_only = bool(args.remove_only) or not replace_with
    if remove_only:
        replace_with = []

    repo_root = Path.cwd()
    paths: set[Path] = set(_iter_default_jsons(dataset_root))
    for pattern in args.include:
        for p in _iter_glob(repo_root, pattern):
            if p.is_file() and p.suffix.lower() == ".json":
                paths.add(p)

    changed: list[Change] = []
    errors = 0
    removed_task_counts = Counter()
    added_task_counts = Counter()
    removed_field_counts = Counter()

    for path in sorted(paths):
        try:
            data = _load_json(path)
        except Exception:
            errors += 1
            continue

        data2, removed_from_tasks_to_annotate, added_to_tasks_to_annotate = (
            _update_tasks_to_annotate(
                data,
                from_tasks=from_tasks,
                replace_with=replace_with,
                remove_only=remove_only,
            )
        )
        data3, removed_fields = remove_task_fields_in_json(
            data=data2,
            tasks=from_tasks,
            remove_keys_and_task_items=bool(args.remove_task_fields),
        )

        if (
            (data3 == data)
            and not removed_from_tasks_to_annotate
            and not added_to_tasks_to_annotate
            and not removed_fields
        ):
            continue

        for t in removed_from_tasks_to_annotate:
            removed_task_counts[t] += 1
        for t in added_to_tasks_to_annotate:
            added_task_counts[t] += 1
        for k in removed_fields:
            removed_field_counts[k] += 1

        changed.append(
            Change(
                path=path,
                removed_tasks_to_annotate=removed_from_tasks_to_annotate,
                added_tasks_to_annotate=added_to_tasks_to_annotate,
                removed_task_fields=removed_fields,
            )
        )

        if args.apply:
            _write_json(path, data3)

    if args.json:
        try:
            for ch in changed:
                print(
                    json.dumps(
                        {
                            "path": str(ch.path),
                            "removed_tasks_to_annotate": list(
                                ch.removed_tasks_to_annotate
                            ),
                            "added_tasks_to_annotate": list(ch.added_tasks_to_annotate),
                            "removed_task_fields": list(ch.removed_task_fields),
                        },
                        ensure_ascii=False,
                    )
                )
        except BrokenPipeError:
            return 1 if changed or errors else 0
    else:
        print("dataset_root:", dataset_root)
        print("tasks:", ",".join(sorted(from_tasks)))
        print("replace_with:", ",".join(replace_with) if replace_with else "(none)")
        print("remove_only:", bool(remove_only))
        print("remove_task_fields:", bool(args.remove_task_fields))
        print("apply:", bool(args.apply))
        print("scanned:", len(paths))
        print("errors:", errors)
        print("changed_files:", len(changed))
        for name, count in sorted(removed_task_counts.items()):
            print(f"removed_from_tasks_to_annotate\t{name}\t{count}")
        for name, count in sorted(added_task_counts.items()):
            print(f"added_to_tasks_to_annotate\t{name}\t{count}")
        for name, count in sorted(removed_field_counts.items()):
            print(f"removed_task_field\t{name}\t{count}")

        if changed and not args.apply:
            print("\nRun with --apply to write changes.")

    return 1 if changed or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
