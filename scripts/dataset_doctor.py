#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Issue:
    path: Path
    reason: str
    details: str


@dataclass(frozen=True)
class Fix:
    path: Path
    reason: str


def _parse_csv_set(value: str) -> set[str]:
    out: set[str] = set()
    for raw in value.split(","):
        name = raw.strip()
        if name:
            out.add(name)
    return out


def _infer_sport_event(dataset_root: Path, path: Path) -> tuple[str, str] | None:
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        return None

    if len(rel.parts) < 3:
        return None

    sport, event = rel.parts[0], rel.parts[1]
    if not sport or not event:
        return None

    return sport, event


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def _iter_segment_metadata_jsons(dataset_root: Path) -> Iterable[Path]:
    yield from sorted(dataset_root.glob("*/*/frames/*.json"))
    yield from sorted(dataset_root.glob("*/*/clips/*.json"))


def _iter_event_level_jsons(dataset_root: Path) -> Iterable[Path]:
    for p in sorted(dataset_root.glob("*/*/*.json")):
        if p.parent.name in ("frames", "clips"):
            continue
        yield p


def _normalize_task(item: Any) -> str | None:
    if not isinstance(item, str):
        return None
    name = item.strip()
    return name if name else None


def _parse_tasks_to_annotate(data: Any) -> tuple[list[str], list[str]]:
    if not isinstance(data, dict):
        return [], ["invalid_json_root"]

    raw = data.get("tasks_to_annotate")
    if raw is None:
        return [], ["missing_tasks_to_annotate"]
    if not isinstance(raw, list):
        return [], ["tasks_to_annotate_not_a_list"]

    issues: list[str] = []
    tasks: list[str] = []

    for item in raw:
        if not isinstance(item, str):
            issues.append("tasks_to_annotate_contains_non_string")
            continue
        name = item.strip()
        if not name:
            issues.append("tasks_to_annotate_contains_empty_string")
            continue
        tasks.append(name)

    if not tasks:
        issues.append("empty_tasks_to_annotate")

    return tasks, sorted(set(issues))


def _fix_sport_event_fields(data: Any, sport: str, event: str) -> tuple[Any, bool]:
    if not isinstance(data, dict):
        return data, False

    changed = False

    if "sport" in data and (
        not isinstance(data.get("sport"), str) or data.get("sport") != sport
    ):
        data["sport"] = sport
        changed = True

    if "event" in data and (
        not isinstance(data.get("event"), str) or data.get("event") != event
    ):
        data["event"] = event
        changed = True

    origin = data.get("origin")
    if isinstance(origin, dict):
        if origin.get("sport") != sport:
            origin["sport"] = sport
            changed = True
        if origin.get("event") != event:
            origin["event"] = event
            changed = True
        data["origin"] = origin

    if ("origin" not in data) and ("id" in data) and ("tasks_to_annotate" in data):
        data["origin"] = {"sport": sport, "event": event}
        changed = True

    return data, changed


def _get_segment_id(data: Any, default: str) -> str:
    if isinstance(data, dict):
        v = data.get("id")
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
    return default


def _get_total_frames(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    info = data.get("info")
    if not isinstance(info, dict):
        return None
    v = info.get("total_frames")
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _expected_media_paths(event_dir: Path, kind: str, seg_id: str) -> list[Path]:
    if kind == "clips":
        return [event_dir / "clips" / f"{seg_id}.mp4"]

    if kind == "frames":
        return [
            event_dir / "frames" / f"{seg_id}.jpg",
            event_dir / "frames" / f"{seg_id}.jpeg",
            event_dir / "frames" / f"{seg_id}.png",
        ]

    return []


def _get_declared_sport_event(data: Any) -> tuple[str | None, str | None]:
    if not isinstance(data, dict):
        return None, None

    origin = data.get("origin")
    if isinstance(origin, dict):
        s = origin.get("sport")
        e = origin.get("event")
        if isinstance(s, str) and isinstance(e, str) and s and e:
            return s, e

    s = data.get("sport")
    e = data.get("event")
    if isinstance(s, str) and isinstance(e, str) and s and e:
        return s, e

    return None, None


def _extract_mot_refs(output_data: Any) -> set[str]:
    refs: set[str] = set()
    if not isinstance(output_data, dict):
        return refs

    annotations = output_data.get("annotations")
    if not isinstance(annotations, list):
        return refs

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        tracking = ann.get("tracking_bboxes")
        if not isinstance(tracking, dict):
            continue
        mot = tracking.get("mot_file")
        if isinstance(mot, str) and mot.strip():
            refs.add(mot.strip())

    return refs


def _resolve_mot_path(project_root: Path, mot_ref: str) -> Path:
    p = Path(mot_ref)
    if p.is_absolute():
        return p
    return project_root / p


def _delete_output_for_segment(
    *,
    output_root: Path,
    project_root: Path,
    sport: str,
    event: str,
    kind: str,
    seg_id: str,
    apply: bool,
) -> tuple[int, int]:
    deleted_outputs = 0
    deleted_mot = 0

    out_json = output_root / sport / event / kind / f"{seg_id}.json"
    mot_dir = output_root / sport / event / kind / "mot"

    mot_refs: set[str] = set()
    if out_json.exists():
        try:
            mot_refs = _extract_mot_refs(_load_json(out_json))
        except Exception:
            mot_refs = set()

        if apply:
            try:
                out_json.unlink()
                deleted_outputs += 1
            except OSError:
                pass
        else:
            deleted_outputs += 1

    if mot_dir.exists():
        fallback = set(str(p) for p in mot_dir.glob(f"{seg_id}_*.txt"))
        for ref in mot_refs:
            fallback.add(str(_resolve_mot_path(project_root, ref)))

        for ref in sorted(fallback):
            mot_path = Path(ref)
            if apply:
                try:
                    if mot_path.exists():
                        mot_path.unlink()
                        deleted_mot += 1
                except OSError:
                    pass
            else:
                if mot_path.exists():
                    deleted_mot += 1

    return deleted_outputs, deleted_mot


def _load_task_catalog() -> set[str] | None:
    try:
        from auto_annotator.annotators.task_annotators import TaskAnnotatorFactory

        return set(TaskAnnotatorFactory.get_available_tasks())
    except Exception:
        return None


def _load_sync_prune_module(path: Path):
    name = f"sync_prune_outputs_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _print_issue_summary(issues: list[Issue], max_list: int, show_paths: bool) -> None:
    counts = Counter(i.reason for i in issues)
    print("issues:", len(issues))
    for reason, count in sorted(counts.items()):
        print(f"issue\t{reason}\t{count}")
        if show_paths:
            paths = [i.path for i in issues if i.reason == reason]
            for p in paths[:max_list]:
                d = next(
                    (x.details for x in issues if x.reason == reason and x.path == p),
                    "",
                )
                if d:
                    print(f"{p}\t{d}")
                else:
                    print(p)
            if len(paths) > max_list:
                print(f"... ({len(paths) - max_list} more)")


def _collect_summary_stats(dataset_root: Path, output_root: Path) -> dict[str, Any]:
    total_tasks = Counter()
    total_clips = 0
    total_frames = 0

    for meta_path in _iter_segment_metadata_jsons(dataset_root):
        try:
            data = _load_json(meta_path)
        except Exception:
            continue

        total_frames_value = _get_total_frames(data)
        if total_frames_value == 1:
            total_frames += 1
        else:
            total_clips += 1

        tasks, _ = _parse_tasks_to_annotate(data)
        for t in tasks:
            total_tasks[t] += 1

    known_tasks = set(total_tasks.keys())

    annotated_tasks = Counter()
    unknown_tasks = Counter()

    annotated_clips_files = 0
    annotated_frames_files = 0
    annotated_clips_with_annotations = 0
    annotated_frames_with_annotations = 0

    if output_root.exists():
        for out_path in list(output_root.glob("**/clips/*.json")) + list(
            output_root.glob("**/frames/*.json")
        ):
            try:
                data = _load_json(out_path)
            except Exception:
                continue

            is_frame = "frames" in out_path.parts
            if is_frame:
                annotated_frames_files += 1
            else:
                annotated_clips_files += 1

            annotations = data.get("annotations") if isinstance(data, dict) else None
            if isinstance(annotations, list) and annotations:
                if is_frame:
                    annotated_frames_with_annotations += 1
                else:
                    annotated_clips_with_annotations += 1

            if isinstance(annotations, list):
                for ann in annotations:
                    if not isinstance(ann, dict):
                        continue
                    task = ann.get("task_L2")
                    if isinstance(task, str) and task:
                        if task in known_tasks:
                            annotated_tasks[task] += 1
                        else:
                            unknown_tasks[task] += 1

    return {
        "dataset": {
            "total_clips": total_clips,
            "total_frames": total_frames,
            "task_totals": dict(total_tasks),
        },
        "output": {
            "annotated_clips_files": annotated_clips_files,
            "annotated_frames_files": annotated_frames_files,
            "annotated_clips_with_annotations": annotated_clips_with_annotations,
            "annotated_frames_with_annotations": annotated_frames_with_annotations,
            "annotated_task_counts": dict(annotated_tasks),
            "unknown_task_counts": dict(unknown_tasks),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Doctor for dataset consistency: validate Dataset JSONs, fix origin sport/event, sync-prune outputs, and print stats."
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
        default=Path("data/output"),
        help="Output root (default: data/output)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write fixes and delete output artifacts (default: dry-run)",
    )
    parser.add_argument(
        "--no-delete-output-on-fix",
        action="store_true",
        help="Do not delete output JSON/MOT files when fixing clips/frames metadata",
    )
    parser.add_argument(
        "--frame-only-tasks",
        default="ScoreboardSingle,Objects_Spatial_Relationships",
        help="Comma-separated tasks allowed in frames/*.json",
    )
    parser.add_argument(
        "--video-only-tasks",
        default=(
            "ScoreboardMultiple,Object_Tracking,Spatial_Temporal_Grounding,"
            "Continuous_Events_Caption,Continuous_Actions_Caption"
        ),
        help="Comma-separated tasks allowed in clips/*.json",
    )
    parser.add_argument(
        "--max-list",
        type=int,
        default=50,
        help="Max paths to print per issue category (default: 50)",
    )
    parser.add_argument(
        "--list-issues",
        action="store_true",
        help="Print problematic JSON paths",
    )
    parser.add_argument(
        "--fix-id-from-filename",
        action="store_true",
        help="If a segment metadata JSON has id != filename stem, rewrite id to match the filename stem",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    output_root: Path = args.output_root
    delete_output_on_fix = not bool(args.no_delete_output_on_fix)

    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    frame_only = _parse_csv_set(args.frame_only_tasks)
    video_only = _parse_csv_set(args.video_only_tasks)

    overlap = frame_only & video_only
    if overlap:
        print(
            "frame/video task sets overlap: " + ",".join(sorted(overlap)),
            file=sys.stderr,
        )
        return 2

    task_catalog = _load_task_catalog()

    issues: list[Issue] = []
    fixes: list[Fix] = []
    outputs_marked = 0
    mot_marked = 0

    segment_paths = [
        p for p in _iter_segment_metadata_jsons(dataset_root) if p.is_file()
    ]
    event_paths = [p for p in _iter_event_level_jsons(dataset_root) if p.is_file()]

    for meta_path in segment_paths:
        se = _infer_sport_event(dataset_root, meta_path)
        if se is None:
            issues.append(Issue(path=meta_path, reason="bad_path", details=""))
            continue
        sport, event = se

        kind = meta_path.parent.name
        event_dir = dataset_root / sport / event

        try:
            data = _load_json(meta_path)
        except Exception:
            issues.append(Issue(path=meta_path, reason="invalid_json", details=""))
            continue

        declared_sport, declared_event = _get_declared_sport_event(data)

        seg_id_before = _get_segment_id(data, default=meta_path.stem)
        seg_id_after = seg_id_before
        id_fixed = False
        if args.fix_id_from_filename and seg_id_before != meta_path.stem:
            if isinstance(data, dict):
                data["id"] = meta_path.stem
                seg_id_after = meta_path.stem
                id_fixed = True
                if args.apply:
                    _write_json(meta_path, data)
                fixes.append(Fix(path=meta_path, reason="fixed_id_from_filename"))

        fixed_data, changed = _fix_sport_event_fields(data, sport=sport, event=event)
        if changed:
            if args.apply:
                _write_json(meta_path, fixed_data)
            fixes.append(Fix(path=meta_path, reason="fixed_sport_event"))

        if (
            delete_output_on_fix
            and kind in ("frames", "clips")
            and (changed or id_fixed)
        ):
            seg_ids: set[str] = {seg_id_after}
            if seg_id_before != seg_id_after:
                seg_ids.add(seg_id_before)

            se_pairs: set[tuple[str, str]] = {(sport, event)}
            if declared_sport is not None and declared_event is not None:
                se_pairs.add((declared_sport, declared_event))

            for s, e in sorted(se_pairs):
                for seg_id in sorted(seg_ids):
                    o, m = _delete_output_for_segment(
                        output_root=output_root,
                        project_root=Path.cwd(),
                        sport=s,
                        event=e,
                        kind=kind,
                        seg_id=seg_id,
                        apply=bool(args.apply),
                    )
                    outputs_marked += o
                    mot_marked += m

        tasks, task_issues = _parse_tasks_to_annotate(fixed_data)
        for ti in task_issues:
            issues.append(Issue(path=meta_path, reason=ti, details=""))

        if kind == "frames":
            disallowed = sorted(set(tasks) - frame_only)
            if disallowed:
                issues.append(
                    Issue(
                        path=meta_path,
                        reason="frame_json_contains_non_frame_only_task",
                        details=",".join(disallowed),
                    )
                )
        elif kind == "clips":
            disallowed = sorted(set(tasks) - video_only)
            if disallowed:
                issues.append(
                    Issue(
                        path=meta_path,
                        reason="video_json_contains_non_video_only_task",
                        details=",".join(disallowed),
                    )
                )
        else:
            issues.append(Issue(path=meta_path, reason="unknown_kind", details=kind))

        if task_catalog is not None:
            unknown = sorted(t for t in set(tasks) if t not in task_catalog)
            if unknown:
                issues.append(
                    Issue(
                        path=meta_path,
                        reason="tasks_to_annotate_contains_unknown_task",
                        details=",".join(unknown),
                    )
                )

        total_frames = _get_total_frames(fixed_data)
        if kind == "frames" and total_frames not in (None, 1):
            issues.append(
                Issue(
                    path=meta_path,
                    reason="frames_dir_total_frames_not_1",
                    details=str(total_frames),
                )
            )
        if kind == "clips" and total_frames in (None, 0, 1):
            issues.append(
                Issue(
                    path=meta_path,
                    reason="clips_dir_total_frames_not_gt_1",
                    details=str(total_frames),
                )
            )

        media_paths = _expected_media_paths(event_dir, kind=kind, seg_id=seg_id_after)
        if media_paths and not any(p.exists() for p in media_paths):
            issues.append(
                Issue(
                    path=meta_path,
                    reason="content_file_missing",
                    details="|".join(str(p) for p in media_paths),
                )
            )

        if seg_id_after != meta_path.stem:
            issues.append(
                Issue(
                    path=meta_path,
                    reason="id_mismatch_filename",
                    details=f"id={seg_id_after} filename={meta_path.stem}",
                )
            )

    for event_json_path in event_paths:
        se = _infer_sport_event(dataset_root, event_json_path)
        if se is None:
            continue
        sport, event = se

        try:
            data = _load_json(event_json_path)
        except Exception:
            issues.append(
                Issue(path=event_json_path, reason="invalid_json", details="")
            )
            continue

        fixed_data, changed = _fix_sport_event_fields(data, sport=sport, event=event)
        if changed:
            if args.apply:
                _write_json(event_json_path, fixed_data)
            fixes.append(Fix(path=event_json_path, reason="fixed_sport_event"))

        if isinstance(fixed_data, dict):
            video_id = fixed_data.get("video_id")
            if isinstance(video_id, str) and video_id.strip():
                p = dataset_root / sport / event / f"{video_id.strip()}.mp4"
                if not p.exists():
                    issues.append(
                        Issue(
                            path=event_json_path,
                            reason="event_video_missing",
                            details=str(p),
                        )
                    )

            annotations = fixed_data.get("annotations")
            if annotations is None:
                issues.append(
                    Issue(
                        path=event_json_path, reason="missing_annotations", details=""
                    )
                )
            elif not isinstance(annotations, list):
                issues.append(
                    Issue(
                        path=event_json_path,
                        reason="annotations_not_a_list",
                        details=type(annotations).__name__,
                    )
                )
            elif not annotations:
                issues.append(
                    Issue(path=event_json_path, reason="no_annotations", details="")
                )

    sync_module_path = Path(__file__).resolve().parent / "sync_prune_outputs.py"
    sync_mod = _load_sync_prune_module(sync_module_path)

    sync_changes, sync_issues, sync_counters = sync_mod.sync_prune(
        dataset_root=dataset_root,
        output_root=output_root,
        project_root=Path.cwd(),
        apply=bool(args.apply),
        prune_orphans=True,
        delete_empty_outputs=True,
    )

    print("Step1: Dataset validation")
    print("segment_jsons:", len(segment_paths))
    print("event_jsons  :", len(event_paths))
    print("fixes        :", len(fixes))
    print("delete_output_on_fix:", bool(delete_output_on_fix))
    if delete_output_on_fix:
        label_outputs = "outputs_deleted" if args.apply else "outputs_would_delete"
        label_mot = "mot_deleted" if args.apply else "mot_would_delete"
        print(f"{label_outputs}: {outputs_marked}")
        print(f"{label_mot}: {mot_marked}")
    _print_issue_summary(
        issues, max_list=args.max_list, show_paths=bool(args.list_issues)
    )

    print()
    print("Step2: Output sync")
    print("sync_changes:", len(sync_changes))
    print("sync_issues :", len(sync_issues))
    for k, v in sorted(sync_counters.items()):
        print(f"{k}\t{v}")

    if args.list_issues and sync_issues:
        counts = Counter(i.reason for i in sync_issues)
        for reason, count in sorted(counts.items()):
            print(f"sync_issue\t{reason}\t{count}")
            paths = [i.path for i in sync_issues if i.reason == reason]
            for p in paths[: args.max_list]:
                print(p)
            if len(paths) > args.max_list:
                print(f"... ({len(paths) - args.max_list} more)")

    print()
    print("Step3: Stats")
    stats = _collect_summary_stats(dataset_root, output_root)
    print("Dataset counts")
    print("  Total clips:", stats["dataset"]["total_clips"])
    print("  Total frames:", stats["dataset"]["total_frames"])

    print("Task totals (from tasks_to_annotate)")
    for task, count in sorted(stats["dataset"]["task_totals"].items()):
        print(f"  {task}: {count}")

    print("Annotated outputs (from output JSON files)")
    print("  Annotated clips (files):", stats["output"]["annotated_clips_files"])
    print("  Annotated frames (files):", stats["output"]["annotated_frames_files"])
    print(
        "  Annotated clips (with annotations):",
        stats["output"]["annotated_clips_with_annotations"],
    )
    print(
        "  Annotated frames (with annotations):",
        stats["output"]["annotated_frames_with_annotations"],
    )

    print("Annotated task counts (from output annotations)")
    for task, count in sorted(stats["output"]["annotated_task_counts"].items()):
        print(f"  {task}: {count}")

    if stats["output"]["unknown_task_counts"]:
        print("Unknown tasks in outputs")
        for task, count in sorted(stats["output"]["unknown_task_counts"].items()):
            print(f"  {task}: {count}")

    exit_code = 0
    if issues or sync_issues:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
