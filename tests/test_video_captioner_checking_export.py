"""Tests for exporting checking subsets from caption_outputs."""

from __future__ import annotations

import json
from pathlib import Path

from video_captioner.checking_export import export_caption_checking


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _write_long_caption(path: Path, *, total_frames: int = 10, fps: float = 5.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "info": {"original_starting_frame": 0, "total_frames": total_frames, "fps": fps},
                "segment_summary": "Summary.",
                "spans": [
                    {"start_frame": 0, "end_frame": 1, "caption": "A", "chunk_index": 0},
                    {"start_frame": 2, "end_frame": 3, "caption": "B", "chunk_index": 0},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_export_caption_checking_copies_completed_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "caption_data" / "Dataset"
    _touch(dataset_root / "SportA" / "EventDone" / "1.mp4")
    _touch(dataset_root / "SportA" / "EventPartial" / "1.mp4")
    _touch(dataset_root / "SportB" / "EventBad" / "1.mp4")

    output_root = tmp_path / "caption_outputs"

    done_dir = output_root / "SportA" / "EventDone"
    done_dir.mkdir(parents=True)
    (done_dir / "segment.mp4").write_bytes(b"segment-v1")
    (done_dir / "run_meta.json").write_text('{"ok": true}\n', encoding="utf-8")
    _write_long_caption(done_dir / "long_caption.json")

    partial_dir = output_root / "SportA" / "EventPartial"
    partial_dir.mkdir(parents=True)
    (partial_dir / "segment.mp4").write_bytes(b"segment-partial")
    (partial_dir / "run_meta.json").write_text('{"partial": true}\n', encoding="utf-8")

    bad_dir = output_root / "SportB" / "EventBad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "long_caption.json").write_text("{not json", encoding="utf-8")

    dest_root = tmp_path / "caption_checking"
    report = export_caption_checking(
        dataset_root=tmp_path / "caption_data",
        output_root=output_root,
        dest_root=dest_root,
        overwrite=False,
        progress=False,
    )

    assert [f"{k.sport}/{k.event}" for k in report.completed_events] == ["SportA/EventDone"]

    copied_dir = dest_root / "SportA" / "EventDone"
    assert (copied_dir / "segment.mp4").read_bytes() == b"segment-v1"
    assert (copied_dir / "run_meta.json").read_text(encoding="utf-8") == '{"ok": true}\n'
    assert (copied_dir / "long_caption.json").is_file()

    assert not (dest_root / "SportA" / "EventPartial").exists()
    assert not (dest_root / "SportB" / "EventBad").exists()

    completed_json = json.loads((dest_root / "completed_events.json").read_text(encoding="utf-8"))
    assert completed_json["completed_events"] == 1
    assert completed_json["events"] == [{"sport": "SportA", "event": "EventDone"}]

    completed_txt = (dest_root / "completed_events.txt").read_text(encoding="utf-8").strip().splitlines()
    assert completed_txt == ["SportA/EventDone"]

    # Overwrite behavior: should skip by default, and update when overwrite=True.
    (done_dir / "segment.mp4").write_bytes(b"segment-v2")
    export_caption_checking(
        dataset_root=tmp_path / "caption_data",
        output_root=output_root,
        dest_root=dest_root,
        overwrite=False,
        progress=False,
    )
    assert (copied_dir / "segment.mp4").read_bytes() == b"segment-v1"

    export_caption_checking(
        dataset_root=tmp_path / "caption_data",
        output_root=output_root,
        dest_root=dest_root,
        overwrite=True,
        progress=False,
    )
    assert (copied_dir / "segment.mp4").read_bytes() == b"segment-v2"

