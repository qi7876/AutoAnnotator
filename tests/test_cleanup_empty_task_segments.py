from __future__ import annotations

import json
from pathlib import Path

from scripts.cleanup_empty_task_segments import cleanup_empty_task_segments


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_cleanup_empty_task_segments_skips_all_empty_event(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"

    keep_event = dataset_root / "SportA" / "EventKeep"
    skip_event = dataset_root / "SportB" / "EventSkip"

    # EventKeep: has at least one non-empty task JSON, so empty ones should be removed.
    _write_json(
        keep_event / "clips" / "seg_keep.json",
        {"id": "seg_keep", "tasks_to_annotate": ["ScoreboardSingle"]},
    )
    (keep_event / "clips" / "seg_keep.mp4").write_bytes(b"keep")

    _write_json(
        keep_event / "clips" / "seg_drop_clip.json",
        {"id": "seg_drop_clip", "task_to_annotate": []},
    )
    (keep_event / "clips" / "seg_drop_clip.mp4").write_bytes(b"drop")

    _write_json(
        keep_event / "frames" / "seg_drop_frame.json",
        {"id": "seg_drop_frame", "tasks_to_annotate": []},
    )
    (keep_event / "frames" / "seg_drop_frame.jpg").write_bytes(b"drop")

    # EventSkip: all JSONs have empty task lists, so entire event should be skipped.
    _write_json(
        skip_event / "clips" / "all_empty_1.json",
        {"id": "all_empty_1", "task_to_annotate": []},
    )
    _write_json(
        skip_event / "frames" / "all_empty_2.json",
        {"id": "all_empty_2", "tasks_to_annotate": []},
    )
    (skip_event / "clips" / "all_empty_1.mp4").write_bytes(b"skip")
    (skip_event / "frames" / "all_empty_2.jpg").write_bytes(b"skip")

    changes, stats = cleanup_empty_task_segments(dataset_root=dataset_root, apply=False)
    assert stats.scanned_events == 2
    assert stats.eligible_events == 1
    assert stats.skipped_all_empty_events == 1
    assert stats.deleted_jsons == 2
    assert stats.deleted_media_files == 2
    assert len(changes) == 2

    # Dry-run should not modify files.
    assert (keep_event / "clips" / "seg_drop_clip.json").exists()
    assert (keep_event / "frames" / "seg_drop_frame.json").exists()

    cleanup_empty_task_segments(dataset_root=dataset_root, apply=True)

    # Empty entries in eligible event are deleted with corresponding media.
    assert not (keep_event / "clips" / "seg_drop_clip.json").exists()
    assert not (keep_event / "clips" / "seg_drop_clip.mp4").exists()
    assert not (keep_event / "frames" / "seg_drop_frame.json").exists()
    assert not (keep_event / "frames" / "seg_drop_frame.jpg").exists()

    # Non-empty entry remains.
    assert (keep_event / "clips" / "seg_keep.json").exists()
    assert (keep_event / "clips" / "seg_keep.mp4").exists()

    # All-empty event remains untouched.
    assert (skip_event / "clips" / "all_empty_1.json").exists()
    assert (skip_event / "frames" / "all_empty_2.json").exists()
    assert (skip_event / "clips" / "all_empty_1.mp4").exists()
    assert (skip_event / "frames" / "all_empty_2.jpg").exists()
