#!/usr/bin/env python3
"""Summarize dataset and annotation counts for clips/frames and tasks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from auto_annotator.adapters import InputAdapter
from auto_annotator.annotators.task_annotators import TaskAnnotatorFactory
from auto_annotator.config import get_config


def _iter_metadata_paths(dataset_root: Path) -> list[Path]:
    paths: list[Path] = []
    paths.extend(dataset_root.glob("*/*/clips/*.json"))
    paths.extend(dataset_root.glob("*/*/frames/*.json"))
    return paths


def _iter_output_paths(output_root: Path) -> list[Path]:
    paths: list[Path] = []
    paths.extend(output_root.glob("**/clips/*.json"))
    paths.extend(output_root.glob("**/frames/*.json"))
    return paths


def main() -> None:
    config = get_config()
    dataset_root = Path(config.dataset_root)
    output_root = Path(config.project_root) / config.output.temp_dir

    tasks = TaskAnnotatorFactory.get_available_tasks()
    total_tasks = Counter({task: 0 for task in tasks})
    annotated_tasks = Counter({task: 0 for task in tasks})
    unknown_tasks = Counter()

    total_clips = 0
    total_frames = 0
    annotated_clips = 0
    annotated_frames = 0
    annotated_clips_with_annotations = 0
    annotated_frames_with_annotations = 0

    for meta_path in _iter_metadata_paths(dataset_root):
        try:
            metadata = InputAdapter.load_from_json(meta_path)
        except Exception as exc:
            print(f"[warn] Failed to load metadata {meta_path}: {exc}")
            continue

        if metadata.info.is_single_frame():
            total_frames += 1
        else:
            total_clips += 1

        for task in metadata.tasks_to_annotate:
            total_tasks[task] += 1

    if output_root.exists():
        for out_path in _iter_output_paths(output_root):
            try:
                data = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[warn] Failed to load output {out_path}: {exc}")
                continue

            if "frames" in out_path.parts:
                annotated_frames += 1
            else:
                annotated_clips += 1

            annotations = data.get("annotations", [])
            if isinstance(annotations, list) and annotations:
                if "frames" in out_path.parts:
                    annotated_frames_with_annotations += 1
                else:
                    annotated_clips_with_annotations += 1

            if isinstance(annotations, list):
                for ann in annotations:
                    if not isinstance(ann, dict):
                        continue
                    task = ann.get("task_L2")
                    if task in total_tasks:
                        annotated_tasks[task] += 1
                    elif task:
                        unknown_tasks[task] += 1

    print("Dataset counts")
    print(f"  Total clips: {total_clips}")
    print(f"  Total frames: {total_frames}")
    print("")
    print("Task totals (from tasks_to_annotate)")
    for task in tasks:
        print(f"  {task}: {total_tasks[task]}")
    if unknown_tasks:
        print("  Unknown tasks:")
        for task, count in unknown_tasks.items():
            print(f"    {task}: {count}")
    print("")
    print("Annotated outputs (from output JSON files)")
    print(f"  Annotated clips (files): {annotated_clips}")
    print(f"  Annotated frames (files): {annotated_frames}")
    print(f"  Annotated clips (with annotations): {annotated_clips_with_annotations}")
    print(f"  Annotated frames (with annotations): {annotated_frames_with_annotations}")
    print("")
    print("Annotated task counts (from output annotations)")
    for task in tasks:
        print(f"  {task}: {annotated_tasks[task]}")


if __name__ == "__main__":
    main()
