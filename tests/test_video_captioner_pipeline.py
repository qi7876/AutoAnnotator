"""Tests for video_captioner.pipeline end-to-end (offline model)."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

from video_captioner.model import FakeCaptionModel
from video_captioner.pipeline import EventVideo, process_event_video
from video_captioner.schema import LongCaptionResponse


def _make_test_video(path: Path, *, duration_sec: float, fps: int = 30) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={fps}",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=44100",
            "-t",
            f"{duration_sec:.3f}",
            "-c:v",
            "libopenh264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(fps),
            "-keyint_min",
            str(fps),
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_process_event_video_writes_all_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    video_dir = dataset_root / "SportA" / "EventA"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "1.mp4"
    _make_test_video(video_path, duration_sec=8.0)

    out_root = tmp_path / "out"
    segment_path, chunk_records, long_record = process_event_video(
        event_video=EventVideo(sport="SportA", event="EventA", video_path=video_path),
        output_root=out_root,
        model=FakeCaptionModel(),
        rng=random.Random(0),
        language="zh",
        segment_min_sec=2.0,
        segment_max_sec=10.0,
        segment_fraction=0.8,
        chunk_sec=3.0,
        overwrite=True,
    )

    event_out = out_root / "SportA" / "EventA"
    assert segment_path == event_out / "segment.mp4"
    assert segment_path.is_file()
    assert (event_out / "chunks").is_dir()
    assert (event_out / "chunk_captions.json").is_file()
    assert (event_out / "long_caption.json").is_file()
    assert (event_out / "run_meta.json").is_file()

    chunk_payload = json.loads((event_out / "chunk_captions.json").read_text(encoding="utf-8"))
    assert len(chunk_payload) == len(chunk_records)
    assert len(chunk_payload) > 0
    assert chunk_payload[0]["spans"]

    long_payload = json.loads((event_out / "long_caption.json").read_text(encoding="utf-8"))
    parsed = LongCaptionResponse.model_validate(long_payload)
    assert parsed.long_caption
    assert long_record.response.long_caption == parsed.long_caption

    # Resume should return existing outputs without error.
    segment_path2, chunk_records2, long_record2 = process_event_video(
        event_video=EventVideo(sport="SportA", event="EventA", video_path=video_path),
        output_root=out_root,
        model=FakeCaptionModel(),
        rng=random.Random(123),
        language="zh",
        segment_min_sec=2.0,
        segment_max_sec=10.0,
        segment_fraction=0.8,
        chunk_sec=3.0,
        overwrite=False,
    )
    assert segment_path2 == segment_path
    assert len(chunk_records2) == len(chunk_records)
    assert long_record2.response.long_caption == long_record.response.long_caption

