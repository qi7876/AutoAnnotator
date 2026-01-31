"""Tests for video_captioner.progress utilities."""

from __future__ import annotations

import json
from pathlib import Path

from video_captioner.progress import collect_progress, format_duration


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_collect_progress_counts_completed_partial_and_todo(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    _touch(dataset_root / "SportA" / "EventA" / "1.mp4")
    _touch(dataset_root / "SportA" / "EventB" / "1.mp4")
    _touch(dataset_root / "SportB" / "EventC" / "1.mp4")

    output_root = tmp_path / "caption_outputs"

    # Completed output (long_caption.json exists and parses).
    event_a_out = output_root / "SportA" / "EventA"
    event_a_out.mkdir(parents=True)
    (event_a_out / "long_caption.json").write_text(
        json.dumps(
            {
                "info": {"original_starting_frame": 0, "total_frames": 600, "fps": 10.0},
                "segment_summary": "Summary.",
                "spans": [
                    {"start_frame": 0, "end_frame": 9, "caption": "A", "chunk_index": 0},
                    {"start_frame": 10, "end_frame": 19, "caption": "B", "chunk_index": 0},
                    {"start_frame": 20, "end_frame": 29, "caption": "C", "chunk_index": 0},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Partial output (chunk_captions.json exists, long_caption.json missing).
    event_b_out = output_root / "SportA" / "EventB"
    event_b_out.mkdir(parents=True)
    (event_b_out / "chunk_captions.json").write_text(
        json.dumps(
            [
                {
                    "chunk_index": 0,
                    "chunk_path": "chunks/chunk_000.mp4",
                    "info": {"original_starting_frame": 0, "total_frames": 300, "fps": 10.0},
                    "chunk_summary": "Chunk summary.",
                    "spans": [
                        {"start_frame": 0, "end_frame": 9, "caption": "X"},
                        {"start_frame": 10, "end_frame": 19, "caption": "Y"},
                    ],
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary, events = collect_progress(dataset_root=dataset_root, output_root=output_root)

    assert summary.total_events == 3
    assert summary.completed_events == 1
    assert summary.partial_events == 1
    assert summary.not_started_events == 1
    assert summary.error_events == 0

    assert abs(summary.completed_duration_sec - 60.0) < 1e-9
    assert summary.completed_span_count == 3

    assert summary.by_sport_total == {"SportA": 2, "SportB": 1}
    assert summary.by_sport_completed == {"SportA": 1}

    # Ensure we return per-event progress records.
    assert len(events) == 3


def test_collect_progress_reports_error_on_bad_long_caption(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    _touch(dataset_root / "SportA" / "EventA" / "1.mp4")

    output_root = tmp_path / "caption_outputs"
    event_out = output_root / "SportA" / "EventA"
    event_out.mkdir(parents=True)
    (event_out / "long_caption.json").write_text("{not json", encoding="utf-8")

    summary, events = collect_progress(dataset_root=dataset_root, output_root=output_root)
    assert summary.total_events == 1
    assert summary.completed_events == 0
    assert summary.error_events == 1
    assert events[0].status == "error"


def test_format_duration() -> None:
    assert format_duration(0) == "0s"
    assert format_duration(59) == "59s"
    assert format_duration(60) == "1m 0s"
    assert format_duration(60 * 60 + 61) == "1h 1m 1s"

