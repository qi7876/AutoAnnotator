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
    output_path: Path
    removed_tasks: tuple[str, ...]
    kept_tasks: tuple[str, ...]
    removed_mot_files: tuple[str, ...]


@dataclass(frozen=True)
class Issue:
    path: Path
    reason: str


def _parse_task_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            name = item.strip()
            if name:
                out.append(name)
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _iter_metadata_paths(dataset_root: Path) -> Iterable[Path]:
    yield from sorted(dataset_root.glob("*/*/frames/*.json"))
    yield from sorted(dataset_root.glob("*/*/clips/*.json"))


def _infer_origin_from_path(
    dataset_root: Path, path: Path
) -> tuple[str | None, str | None, str | None]:
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        return None, None, None

    if len(rel.parts) < 4:
        return None, None, None

    sport, event, kind = rel.parts[0], rel.parts[1], rel.parts[2]
    if kind not in ("frames", "clips"):
        return sport, event, None
    return sport, event, kind


def _get_id_from_metadata(path: Path, data: Any) -> str:
    if isinstance(data, dict):
        v = data.get("id")
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, int):
            return str(v)
    return path.stem


def _extract_mot_files(annotation: Any) -> list[str]:
    if not isinstance(annotation, dict):
        return []
    tracking = annotation.get("tracking_bboxes")
    if not isinstance(tracking, dict):
        return []
    mot_file = tracking.get("mot_file")
    if isinstance(mot_file, str) and mot_file.strip():
        return [mot_file.strip()]
    return []


def _resolve_mot_path(project_root: Path, mot_ref: str) -> Path:
    p = Path(mot_ref)
    if p.is_absolute():
        return p
    return project_root / p


def _unique_in_order(items: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return tuple(out)


def sync_prune(
    *,
    dataset_root: Path,
    output_root: Path,
    project_root: Path,
    apply: bool,
    prune_orphans: bool,
    delete_empty_outputs: bool,
) -> tuple[list[Change], list[Issue], dict[str, int]]:
    counters: Counter[str] = Counter()
    changes: list[Change] = []
    issues: list[Issue] = []

    meta_map: dict[tuple[str, str, str, str], set[str]] = {}
    metadata_paths = list(_iter_metadata_paths(dataset_root))
    counters["metadata_files"] = len(metadata_paths)

    for meta_path in metadata_paths:
        sport, event, kind = _infer_origin_from_path(dataset_root, meta_path)
        if not sport or not event or not kind:
            counters["metadata_skipped_bad_path"] += 1
            issues.append(Issue(path=meta_path, reason="metadata_skipped_bad_path"))
            continue

        try:
            data = _load_json(meta_path)
        except Exception:
            counters["metadata_load_errors"] += 1
            issues.append(Issue(path=meta_path, reason="metadata_load_error"))
            continue

        seg_id = _get_id_from_metadata(meta_path, data)
        if isinstance(data, dict):
            requested = set(_parse_task_list(data.get("tasks_to_annotate")))
        else:
            requested = set()

        meta_map[(sport, event, kind, seg_id)] = requested

    processed_outputs: set[Path] = set()

    for (sport, event, kind, seg_id), requested in sorted(meta_map.items()):
        out_path = output_root / sport / event / kind / f"{seg_id}.json"
        if not out_path.exists():
            counters["output_missing"] += 1
            issues.append(Issue(path=out_path, reason="output_missing"))
            continue

        processed_outputs.add(out_path)

        try:
            out_data = _load_json(out_path)
        except Exception:
            counters["output_load_errors"] += 1
            issues.append(Issue(path=out_path, reason="output_load_error"))
            continue

        if not isinstance(out_data, dict):
            counters["output_invalid_json"] += 1
            issues.append(Issue(path=out_path, reason="output_invalid_json"))
            continue

        anns = out_data.get("annotations")
        if not isinstance(anns, list):
            counters["output_missing_annotations"] += 1
            issues.append(Issue(path=out_path, reason="output_missing_annotations"))
            continue

        kept: list[Any] = []
        removed: list[Any] = []
        kept_tasks_list: list[str] = []
        removed_tasks_list: list[str] = []

        kept_mot_refs: set[str] = set()
        removed_mot_refs: set[str] = set()
        saw_missing_task_l2 = False

        for ann in anns:
            task = ann.get("task_L2") if isinstance(ann, dict) else None
            task_name = task.strip() if isinstance(task, str) else None

            if task_name and task_name in requested:
                kept.append(ann)
                kept_tasks_list.append(task_name)
                for ref in _extract_mot_files(ann):
                    kept_mot_refs.add(ref)
            else:
                removed.append(ann)
                if task_name:
                    removed_tasks_list.append(task_name)
                else:
                    counters["annotation_missing_task_L2"] += 1
                    saw_missing_task_l2 = True
                for ref in _extract_mot_files(ann):
                    removed_mot_refs.add(ref)

        if saw_missing_task_l2:
            issues.append(Issue(path=out_path, reason="annotation_missing_task_L2"))

        if not removed and len(kept) == len(anns):
            continue

        removed_mot_to_delete = sorted(removed_mot_refs - kept_mot_refs)

        new_out_data = dict(out_data)
        new_out_data["annotations"] = kept

        if delete_empty_outputs and not kept:
            if apply:
                try:
                    out_path.unlink()
                    counters["output_deleted_empty"] += 1
                except Exception:
                    counters["output_delete_errors"] += 1
            else:
                counters["output_would_delete_empty"] += 1
        else:
            if apply:
                _write_json(out_path, new_out_data)
                counters["output_rewritten"] += 1
            else:
                counters["output_would_rewrite"] += 1

        deleted_mot: list[str] = []
        for mot_ref in removed_mot_to_delete:
            mot_path = _resolve_mot_path(project_root, mot_ref)
            if apply:
                try:
                    if mot_path.exists():
                        mot_path.unlink()
                        deleted_mot.append(str(mot_path))
                        counters["mot_deleted"] += 1
                    else:
                        counters["mot_missing"] += 1
                except Exception:
                    counters["mot_delete_errors"] += 1
            else:
                if mot_path.exists():
                    counters["mot_would_delete"] += 1
                else:
                    counters["mot_missing"] += 1

        changes.append(
            Change(
                output_path=out_path,
                removed_tasks=_unique_in_order(removed_tasks_list),
                kept_tasks=_unique_in_order(kept_tasks_list),
                removed_mot_files=tuple(deleted_mot)
                if apply
                else tuple(
                    str(_resolve_mot_path(project_root, r))
                    for r in removed_mot_to_delete
                ),
            )
        )

    if prune_orphans and output_root.exists():
        orphan_paths = list(output_root.glob("*/*/frames/*.json")) + list(
            output_root.glob("*/*/clips/*.json")
        )
        for out_path in sorted(orphan_paths):
            if out_path in processed_outputs:
                continue

            try:
                out_data = _load_json(out_path)
            except Exception:
                counters["orphan_output_load_errors"] += 1
                issues.append(Issue(path=out_path, reason="orphan_output_load_error"))
                continue

            mot_refs: set[str] = set()
            if isinstance(out_data, dict) and isinstance(
                out_data.get("annotations"), list
            ):
                for ann in out_data["annotations"]:
                    for ref in _extract_mot_files(ann):
                        mot_refs.add(ref)

            if apply:
                try:
                    out_path.unlink()
                    counters["orphan_output_deleted"] += 1
                except Exception:
                    counters["orphan_output_delete_errors"] += 1

                for ref in mot_refs:
                    p = _resolve_mot_path(project_root, ref)
                    try:
                        if p.exists():
                            p.unlink()
                            counters["mot_deleted"] += 1
                    except Exception:
                        counters["mot_delete_errors"] += 1
            else:
                counters["orphan_output_would_delete"] += 1

    return changes, issues, dict(counters)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prune output annotations so outputs strictly match current tasks_to_annotate in dataset metadata."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root directory containing metadata JSONs (default: data/Dataset)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/output"),
        help="Output root directory containing generated outputs (default: data/output)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve relative mot_file paths (default: cwd)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default: dry-run)",
    )
    parser.add_argument(
        "--prune-orphans",
        action="store_true",
        help="Also delete output JSONs whose segment metadata no longer exists",
    )
    parser.add_argument(
        "--delete-empty-outputs",
        action="store_true",
        help="Delete output JSON if no annotations remain after pruning",
    )
    parser.add_argument(
        "--list-changes",
        action="store_true",
        help="Print each output JSON that would be changed",
    )
    parser.add_argument(
        "--list-issues",
        action="store_true",
        help="Print metadata/output JSON paths with errors (missing, invalid JSON, etc.)",
    )
    parser.add_argument(
        "--max-list",
        type=int,
        default=50,
        help="Max paths to print per issue category (default: 50)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print per-file changes as JSON lines",
    )
    parser.add_argument(
        "--json-issues",
        action="store_true",
        help="Print issues as JSON lines",
    )
    parser.add_argument(
        "--json-all",
        action="store_true",
        help="Print both changes and issues as JSON lines",
    )

    args = parser.parse_args()

    json_mode_count = (
        int(bool(args.json)) + int(bool(args.json_issues)) + int(bool(args.json_all))
    )
    if json_mode_count > 1:
        print(
            "--json, --json-issues, --json-all are mutually exclusive", file=sys.stderr
        )
        return 2

    if args.max_list is not None and args.max_list < 0:
        print("--max-list must be >= 0", file=sys.stderr)
        return 2

    dataset_root: Path = args.dataset_root
    output_root: Path = args.output_root
    project_root: Path = args.project_root

    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    changes, issues, counters = sync_prune(
        dataset_root=dataset_root,
        output_root=output_root,
        project_root=project_root,
        apply=bool(args.apply),
        prune_orphans=bool(args.prune_orphans),
        delete_empty_outputs=bool(args.delete_empty_outputs),
    )

    if args.json_all:
        try:
            for it in issues:
                print(
                    json.dumps(
                        {
                            "record_type": "issue",
                            "path": str(it.path),
                            "reason": it.reason,
                        },
                        ensure_ascii=False,
                    )
                )
            for ch in changes:
                print(
                    json.dumps(
                        {
                            "record_type": "change",
                            "output_path": str(ch.output_path),
                            "removed_tasks": list(ch.removed_tasks),
                            "kept_tasks": list(ch.kept_tasks),
                            "removed_mot_files": list(ch.removed_mot_files),
                        },
                        ensure_ascii=False,
                    )
                )
        except BrokenPipeError:
            return 1 if changes or issues else 0
    elif args.json_issues:
        try:
            for it in issues:
                print(
                    json.dumps(
                        {
                            "path": str(it.path),
                            "reason": it.reason,
                        },
                        ensure_ascii=False,
                    )
                )
        except BrokenPipeError:
            return 1 if issues else 0
    elif args.json:
        try:
            for ch in changes:
                print(
                    json.dumps(
                        {
                            "output_path": str(ch.output_path),
                            "removed_tasks": list(ch.removed_tasks),
                            "kept_tasks": list(ch.kept_tasks),
                            "removed_mot_files": list(ch.removed_mot_files),
                        },
                        ensure_ascii=False,
                    )
                )
        except BrokenPipeError:
            return 1 if changes else 0
    else:
        print("dataset_root:", dataset_root)
        print("output_root :", output_root)
        print("project_root:", project_root)
        print("apply:", bool(args.apply))
        print("prune_orphans:", bool(args.prune_orphans))
        print("delete_empty_outputs:", bool(args.delete_empty_outputs))
        print("changed_files:", len(changes))
        print("issue_count:", len(issues))
        for k, v in sorted(counters.items()):
            print(f"{k}\t{v}")

        if args.list_changes and changes:
            for ch in changes:
                removed = ",".join(ch.removed_tasks) if ch.removed_tasks else "-"
                print(f"change\t{removed}\t{ch.output_path}")

        if args.list_issues and issues:
            grouped: dict[str, list[Path]] = {}
            for it in issues:
                grouped.setdefault(it.reason, []).append(it.path)

            for reason in sorted(grouped.keys()):
                paths = grouped[reason]
                print(f"issue\t{reason}\t{len(paths)}")
                limit = args.max_list
                for p in paths[:limit]:
                    print(p)
                if len(paths) > limit:
                    print(f"... ({len(paths) - limit} more)")

        if (changes or issues) and not args.apply:
            print("\nRun with --apply to write changes.")

    return 1 if (changes or issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
