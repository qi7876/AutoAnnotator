#!/usr/bin/env python3
"""Check for clips with multiple MOT tasks and naming collisions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from auto_annotator.config import get_config


def _iter_output_json(output_root: Path) -> list[Path]:
    return list(output_root.glob("**/clips/*.json"))


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _task_has_mot(annotation: dict) -> bool:
    if annotation.get("mot_file"):
        return True
    tracking = annotation.get("tracking_bboxes")
    if isinstance(tracking, dict) and tracking.get("mot_file"):
        return True
    if tracking:
        return True
    return False


def main() -> None:
    config = get_config()
    output_root = Path(config.project_root) / config.output.temp_dir

    clip_tasks: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    missing_task_suffix: list[str] = []

    for json_path in _iter_output_json(output_root):
        data = _load_json(json_path)
        if not data or not isinstance(data, dict):
            continue
        annotations = data.get("annotations", [])
        if not isinstance(annotations, list):
            continue

        # infer ids from path: output_root/sport/event/clips/{id}.json
        try:
            rel = json_path.relative_to(output_root)
            sport, event, _, clip_file = rel.parts[-4:]
            clip_id = Path(clip_file).stem
        except Exception:
            continue

        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if not _task_has_mot(ann):
                continue
            task_name = ann.get("task_L2", "unknown")
            mot_ref = ann.get("mot_file")
            if not mot_ref and isinstance(ann.get("tracking_bboxes"), dict):
                mot_ref = ann["tracking_bboxes"].get("mot_file")
            mot_ref = mot_ref or ""
            clip_tasks[(sport, event, clip_id)].append((task_name, str(mot_ref)))

            if mot_ref:
                mot_name = Path(mot_ref).name
                expected_suffix = f"_{task_name}"
                if expected_suffix not in mot_name:
                    missing_task_suffix.append(
                        f"{sport}/{event}/{clip_id}: {mot_name} (task={task_name})"
                    )

    multi_task = {
        key: tasks
        for key, tasks in clip_tasks.items()
        if len({t[0] for t in tasks}) > 1
    }
    for (sport, event, clip_id), tasks in sorted(multi_task.items()):
        task_list = ", ".join(sorted({t[0] for t in tasks}))
        print(f"multi-task: {sport}/{event}/{clip_id}: {task_list}")

    for item in missing_task_suffix:
        print(f"missing-suffix: {item}")


if __name__ == "__main__":
    main()
