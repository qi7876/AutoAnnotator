"""Tests for video_captioner.ffmpeg_utils."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

from video_captioner.ffmpeg_utils import (
    keyframe_trim_copy,
    probe_video,
    select_random_segment,
    split_into_chunks,
)


def _make_test_video(path: Path, *, duration_sec: float, fps: int = 30) -> None:
    # GOP=FPS -> keyframe every 1 second; stable for keyframe-only trimming tests.
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


def test_select_random_segment_uses_full_when_short() -> None:
    rng = random.Random(0)
    start, dur = select_random_segment(
        4.0, rng=rng, min_duration_sec=5.0, max_duration_sec=30.0, fraction_of_total=0.8
    )
    assert start == 0.0
    assert dur == 4.0


def test_select_random_segment_duration_scales_and_start_in_range() -> None:
    rng = random.Random(123)
    start, dur = select_random_segment(
        20.0, rng=rng, min_duration_sec=5.0, max_duration_sec=30.0, fraction_of_total=0.8
    )
    assert dur == 16.0
    assert 0.0 <= start <= 4.0


def test_keyframe_trim_and_split(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _make_test_video(src, duration_sec=12.0)

    src_probe = probe_video(src)
    assert src_probe.duration_sec > 0
    assert src_probe.fps is not None
    assert src_probe.total_frames is not None and src_probe.total_frames > 0

    trimmed = tmp_path / "trimmed.mp4"
    keyframe_trim_copy(
        input_path=src,
        output_path=trimmed,
        start_sec=2.0,
        duration_sec=5.0,
        overwrite=True,
    )
    assert trimmed.is_file()

    trimmed_probe = probe_video(trimmed)
    # Stream-copy + keyframe seeking can shift slightly; keep a wide bound but non-vacuous.
    assert 3.5 <= trimmed_probe.duration_sec <= 6.5

    chunks_dir = tmp_path / "chunks"
    chunks = split_into_chunks(
        input_path=trimmed,
        output_dir=chunks_dir,
        chunk_duration_sec=2.0,
        overwrite=True,
    )
    assert len(chunks) >= 2
    assert chunks[0].path.is_file()
    assert chunks[0].index == 0
    assert chunks[0].start_sec == 0.0

    # Ensure each chunk exists and has non-zero duration.
    for chunk in chunks:
        assert chunk.path.is_file()
        c_probe = probe_video(chunk.path)
        assert c_probe.duration_sec > 0


def test_preserve_timestamps_allows_nested_splitting(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _make_test_video(src, duration_sec=12.0)

    seg = tmp_path / "seg.mp4"
    keyframe_trim_copy(
        input_path=src,
        output_path=seg,
        start_sec=5.0,
        duration_sec=4.0,
        overwrite=True,
        preserve_timestamps=True,
    )
    seg_probe = probe_video(seg)
    assert seg_probe.start_time_sec > 0
    assert seg_probe.duration_sec > 0

    chunks = split_into_chunks(
        input_path=seg,
        output_dir=tmp_path / "chunks_preserve",
        chunk_duration_sec=2.0,
        overwrite=True,
        preserve_timestamps=True,
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        c_probe = probe_video(chunk.path)
        assert c_probe.duration_sec > 0
        assert c_probe.start_time_sec >= seg_probe.start_time_sec - 0.1
