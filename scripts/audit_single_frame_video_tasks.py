#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Finding:
    path: Path
    sport: str | None
    event: str | None
    kind: str | None
    total_frames: int | None
    tasks_to_annotate: tuple[str, ...]
    offending_tasks: tuple[str, ...]
    reason: str


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_segment_jsons(dataset_root: Path) -> Iterable[Path]:
    for kind in ("frames", "clips"):
        yield from sorted(dataset_root.glob(f"*/*/{kind}/*.json"))


def _infer_origin(
    dataset_root: Path, path: Path
) -> tuple[str | None, str | None, str | None]:
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        return None, None, None

    parts = rel.parts
    if len(parts) < 4:
        return None, None, None

    sport, event, kind = parts[0], parts[1], parts[2]
    return sport, event, kind


def _normalize_task_name(task: Any) -> str | None:
    if isinstance(task, str):
        cleaned = task.strip()
        return cleaned if cleaned else None
    return None


def _collect_task_mentions(value: Any, targets: set[str], found: set[str]) -> None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned in targets:
            found.add(cleaned)
        return

    if isinstance(value, list):
        for item in value:
            _collect_task_mentions(item, targets, found)
        return

    if isinstance(value, dict):
        for item in value.values():
            _collect_task_mentions(item, targets, found)
        return


def audit_dataset(
    *,
    dataset_root: Path,
    segment_paths: list[Path],
    frame_only_tasks: set[str],
    video_only_tasks: set[str],
    deep_scan: bool,
) -> list[Finding]:
    findings: list[Finding] = []

    for json_path in segment_paths:
        sport, event, kind = _infer_origin(dataset_root, json_path)
        data = _load_json(json_path)
        if not isinstance(data, dict):
            findings.append(
                Finding(
                    path=json_path,
                    sport=sport,
                    event=event,
                    kind=kind,
                    total_frames=None,
                    tasks_to_annotate=(),
                    offending_tasks=(),
                    reason="invalid_json",
                )
            )
            continue

        info = data.get("info")
        if not isinstance(info, dict):
            info = {}
        total_frames = _parse_int(info.get("total_frames"))

        tasks_raw = data.get("tasks_to_annotate")
        if tasks_raw is None:
            tasks = ()
            findings.append(
                Finding(
                    path=json_path,
                    sport=sport,
                    event=event,
                    kind=kind,
                    total_frames=total_frames,
                    tasks_to_annotate=(),
                    offending_tasks=(),
                    reason="missing_tasks_to_annotate",
                )
            )
        elif isinstance(tasks_raw, list):
            has_non_string = any(not isinstance(item, str) for item in tasks_raw)
            has_empty_string = any(
                isinstance(item, str) and not item.strip() for item in tasks_raw
            )

            if has_non_string:
                findings.append(
                    Finding(
                        path=json_path,
                        sport=sport,
                        event=event,
                        kind=kind,
                        total_frames=total_frames,
                        tasks_to_annotate=(),
                        offending_tasks=(),
                        reason="tasks_to_annotate_contains_non_string",
                    )
                )

            if has_empty_string:
                findings.append(
                    Finding(
                        path=json_path,
                        sport=sport,
                        event=event,
                        kind=kind,
                        total_frames=total_frames,
                        tasks_to_annotate=(),
                        offending_tasks=(),
                        reason="tasks_to_annotate_contains_empty_string",
                    )
                )

            tasks = tuple(
                t
                for t in (_normalize_task_name(item) for item in tasks_raw)
                if t is not None
            )

            if not tasks:
                findings.append(
                    Finding(
                        path=json_path,
                        sport=sport,
                        event=event,
                        kind=kind,
                        total_frames=total_frames,
                        tasks_to_annotate=(),
                        offending_tasks=(),
                        reason="empty_tasks_to_annotate",
                    )
                )
        else:
            tasks = ()
            findings.append(
                Finding(
                    path=json_path,
                    sport=sport,
                    event=event,
                    kind=kind,
                    total_frames=total_frames,
                    tasks_to_annotate=(),
                    offending_tasks=(),
                    reason="tasks_to_annotate_not_a_list",
                )
            )

        if kind == "frames" and total_frames not in (None, 1):
            findings.append(
                Finding(
                    path=json_path,
                    sport=sport,
                    event=event,
                    kind=kind,
                    total_frames=total_frames,
                    tasks_to_annotate=tasks,
                    offending_tasks=(),
                    reason="frames_dir_total_frames_not_1",
                )
            )

        if kind == "clips" and total_frames in (None, 0, 1):
            findings.append(
                Finding(
                    path=json_path,
                    sport=sport,
                    event=event,
                    kind=kind,
                    total_frames=total_frames,
                    tasks_to_annotate=tasks,
                    offending_tasks=(),
                    reason="clips_dir_total_frames_not_gt_1",
                )
            )

        if kind == "frames":
            task_set = set(tasks)

            video_only_in_frame = tuple(sorted(task_set & video_only_tasks))
            if video_only_in_frame:
                findings.append(
                    Finding(
                        path=json_path,
                        sport=sport,
                        event=event,
                        kind=kind,
                        total_frames=total_frames,
                        tasks_to_annotate=tasks,
                        offending_tasks=video_only_in_frame,
                        reason="frame_json_contains_video_only_task",
                    )
                )

            disallowed_other = tuple(
                sorted(task_set - frame_only_tasks - video_only_tasks)
            )
            if disallowed_other:
                findings.append(
                    Finding(
                        path=json_path,
                        sport=sport,
                        event=event,
                        kind=kind,
                        total_frames=total_frames,
                        tasks_to_annotate=tasks,
                        offending_tasks=disallowed_other,
                        reason="frame_json_contains_non_frame_only_task",
                    )
                )

            if deep_scan:
                mentioned: set[str] = set()
                _collect_task_mentions(data, video_only_tasks, mentioned)
                if mentioned:
                    findings.append(
                        Finding(
                            path=json_path,
                            sport=sport,
                            event=event,
                            kind=kind,
                            total_frames=total_frames,
                            tasks_to_annotate=tasks,
                            offending_tasks=tuple(sorted(mentioned)),
                            reason="frame_json_mentions_video_only_task",
                        )
                    )

        if kind == "clips":
            task_set = set(tasks)

            frame_only_in_video = tuple(sorted(task_set & frame_only_tasks))
            if frame_only_in_video:
                findings.append(
                    Finding(
                        path=json_path,
                        sport=sport,
                        event=event,
                        kind=kind,
                        total_frames=total_frames,
                        tasks_to_annotate=tasks,
                        offending_tasks=frame_only_in_video,
                        reason="video_json_contains_frame_only_task",
                    )
                )

            disallowed_other = tuple(
                sorted(task_set - video_only_tasks - frame_only_tasks)
            )
            if disallowed_other:
                findings.append(
                    Finding(
                        path=json_path,
                        sport=sport,
                        event=event,
                        kind=kind,
                        total_frames=total_frames,
                        tasks_to_annotate=tasks,
                        offending_tasks=disallowed_other,
                        reason="video_json_contains_non_video_only_task",
                    )
                )

            if deep_scan:
                mentioned = set()
                _collect_task_mentions(data, frame_only_tasks, mentioned)
                if mentioned:
                    findings.append(
                        Finding(
                            path=json_path,
                            sport=sport,
                            event=event,
                            kind=kind,
                            total_frames=total_frames,
                            tasks_to_annotate=tasks,
                            offending_tasks=tuple(sorted(mentioned)),
                            reason="video_json_mentions_frame_only_task",
                        )
                    )

    return findings


def _parse_csv_set(value: str) -> set[str]:
    items = [v.strip() for v in value.split(",")]
    return {v for v in items if v}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit data/Dataset segment JSONs for task allowlist mismatches: "
            "frames JSONs must only contain frame-only tasks, and clips JSONs must only contain video-only tasks."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root directory (default: data/Dataset)",
    )
    parser.add_argument(
        "--frame-only-tasks",
        default="ScoreboardSingle,Objects_Spatial_Relationships",
        help=(
            "Comma-separated task names allowed in frame JSONs under */frames/*.json "
            "(default: ScoreboardSingle,Objects_Spatial_Relationships)"
        ),
    )
    parser.add_argument(
        "--video-only-tasks",
        default=(
            "ScoreboardMultiple,Object_Tracking,Spatial_Temporal_Grounding,"
            "Continuous_Events_Caption,Continuous_Actions_Caption"
        ),
        help=(
            "Comma-separated task names allowed in video JSONs under */clips/*.json "
            "(default: ScoreboardMultiple,Object_Tracking,Spatial_Temporal_Grounding,"
            "Continuous_Events_Caption,Continuous_Actions_Caption)"
        ),
    )
    parser.add_argument(
        "--deep-scan",
        action="store_true",
        help=(
            "Also scan entire JSON objects for task-name strings (can be noisy; default: off)"
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print counts per reason only",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print findings as JSON lines (one per finding)",
    )

    args = parser.parse_args()

    if args.summary and args.json:
        print("--summary and --json cannot be used together", file=sys.stderr)
        return 2

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    segment_paths = list(_iter_segment_jsons(dataset_root))
    frame_only_tasks = _parse_csv_set(args.frame_only_tasks)
    video_only_tasks = _parse_csv_set(args.video_only_tasks)

    overlap = frame_only_tasks & video_only_tasks
    if overlap:
        print(
            "frame-only tasks and video-only tasks overlap: "
            + ",".join(sorted(overlap)),
            file=sys.stderr,
        )
        return 2

    findings = audit_dataset(
        dataset_root=dataset_root,
        segment_paths=segment_paths,
        frame_only_tasks=frame_only_tasks,
        video_only_tasks=video_only_tasks,
        deep_scan=args.deep_scan,
    )

    if args.summary:
        from collections import Counter

        counts = Counter(f.reason for f in findings)
        print("dataset_root:", dataset_root)
        print(
            "frame_only_tasks:",
            ",".join(sorted(frame_only_tasks)) if frame_only_tasks else "(none)",
        )
        print(
            "video_only_tasks:",
            ",".join(sorted(video_only_tasks)) if video_only_tasks else "(none)",
        )
        print("deep_scan:", bool(args.deep_scan))
        print("scanned:", len(segment_paths))
        print("findings:", len(findings))
        for reason, count in sorted(counts.items()):
            print(f"{reason}\t{count}")
        return 1 if findings else 0

    if args.json:
        try:
            for f in findings:
                print(
                    json.dumps(
                        {
                            "path": str(f.path),
                            "sport": f.sport,
                            "event": f.event,
                            "kind": f.kind,
                            "total_frames": f.total_frames,
                            "tasks_to_annotate": list(f.tasks_to_annotate),
                            "offending_tasks": list(f.offending_tasks),
                            "reason": f.reason,
                        },
                        ensure_ascii=False,
                    )
                )
        except BrokenPipeError:
            return 1 if findings else 0
    else:
        print("dataset_root:", dataset_root)
        print(
            "frame_only_tasks:",
            ",".join(sorted(frame_only_tasks)) if frame_only_tasks else "(none)",
        )
        print(
            "video_only_tasks:",
            ",".join(sorted(video_only_tasks)) if video_only_tasks else "(none)",
        )
        print("deep_scan:", bool(args.deep_scan))
        print("scanned:", len(segment_paths))
        print("findings:", len(findings))

        if findings:
            for f in findings:
                tasks_str = ",".join(f.offending_tasks) if f.offending_tasks else "-"
                tf = f.total_frames if f.total_frames is not None else "-"
                print(f"{f.reason}\t{tf}\t{tasks_str}\t{f.path}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
