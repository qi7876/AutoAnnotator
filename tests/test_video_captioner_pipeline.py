"""Tests for video_captioner.pipeline end-to-end (offline model)."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

from video_captioner.ffmpeg_utils import keyframe_trim_copy, probe_video
from video_captioner.model import FakeCaptionModel
from video_captioner.pipeline import EventVideo, process_event_video
from video_captioner.schema import DenseSegmentCaptionResponse


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
    assert chunk_payload[0]["info"]["original_starting_frame"] >= 0
    assert chunk_payload[0]["info"]["total_frames"] > 0
    assert chunk_payload[0]["info"]["fps"] > 0

    meta_payload = json.loads((event_out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta_payload["segment_info"]["original_starting_frame"] >= 0
    assert meta_payload["segment_info"]["total_frames"] > 0
    assert meta_payload["segment_info"]["fps"] > 0

    long_payload = json.loads((event_out / "long_caption.json").read_text(encoding="utf-8"))
    parsed = DenseSegmentCaptionResponse.model_validate(long_payload)
    assert parsed.segment_summary
    assert len(parsed.spans) > 0
    assert long_record.response.segment_summary == parsed.segment_summary

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
    assert long_record2.response.segment_summary == long_record.response.segment_summary


def test_process_event_video_resumes_after_interruption(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    video_dir = dataset_root / "SportA" / "EventA"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "1.mp4"
    _make_test_video(video_path, duration_sec=8.0)

    class _FailingModel(FakeCaptionModel):
        def __init__(self, *, fail_after: int) -> None:
            super().__init__()
            self._calls = 0
            self._fail_after = fail_after

        def caption_chunk(self, **kwargs):
            self._calls += 1
            if self._calls > self._fail_after:
                raise RuntimeError("simulated crash")
            return super().caption_chunk(**kwargs)

    out_root = tmp_path / "out"
    # First run: crash after the first chunk. Should still persist partial chunk_captions.json.
    try:
        process_event_video(
            event_video=EventVideo(sport="SportA", event="EventA", video_path=video_path),
            output_root=out_root,
            model=_FailingModel(fail_after=1),
            rng=random.Random(0),
            language="zh",
            segment_min_sec=2.0,
            segment_max_sec=10.0,
            segment_fraction=0.8,
            chunk_sec=3.0,
            overwrite=True,
        )
    except RuntimeError:
        pass

    event_out = out_root / "SportA" / "EventA"
    assert (event_out / "segment.mp4").is_file()
    assert (event_out / "chunks").is_dir()
    assert (event_out / "chunk_captions.json").is_file()
    assert not (event_out / "long_caption.json").exists()

    partial_payload = json.loads((event_out / "chunk_captions.json").read_text(encoding="utf-8"))
    assert len(partial_payload) == 1
    assert partial_payload[0]["chunk_index"] == 0

    # Second run: resume from the existing segment/chunks and finish.
    segment_path, chunk_records, _ = process_event_video(
        event_video=EventVideo(sport="SportA", event="EventA", video_path=video_path),
        output_root=out_root,
        model=FakeCaptionModel(),
        rng=random.Random(999),
        language="zh",
        segment_min_sec=2.0,
        segment_max_sec=10.0,
        segment_fraction=0.8,
        chunk_sec=3.0,
        overwrite=False,
    )
    assert segment_path.is_file()
    assert len(chunk_records) >= 2
    assert (event_out / "long_caption.json").is_file()

    payload = json.loads((event_out / "chunk_captions.json").read_text(encoding="utf-8"))
    assert {int(item["chunk_index"]) for item in payload} == set(range(len(payload)))


def test_process_event_video_recovers_when_segment_too_short_vs_meta(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    video_dir = dataset_root / "SportA" / "EventA"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "1.mp4"
    _make_test_video(video_path, duration_sec=10.0)

    out_root = tmp_path / "out"
    segment_path, _, _ = process_event_video(
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
    meta = json.loads((event_out / "run_meta.json").read_text(encoding="utf-8"))
    target = float(meta["segment_duration_sec_target"])

    # Replace segment.mp4 with a much shorter (but valid) clip to simulate an incomplete segment.
    short_seg = event_out / "segment_short.mp4"
    keyframe_trim_copy(
        input_path=segment_path,
        output_path=short_seg,
        start_sec=0.0,
        duration_sec=1.0,
        overwrite=True,
        preserve_timestamps=True,
    )
    short_seg.replace(segment_path)

    assert probe_video(segment_path).duration_sec < target * 0.5

    # Resume should detect mismatch vs recorded target and re-extract the segment.
    process_event_video(
        event_video=EventVideo(sport="SportA", event="EventA", video_path=video_path),
        output_root=out_root,
        model=FakeCaptionModel(),
        rng=random.Random(999),
        language="zh",
        segment_min_sec=2.0,
        segment_max_sec=10.0,
        segment_fraction=0.8,
        chunk_sec=3.0,
        overwrite=False,
    )
    assert probe_video(segment_path).duration_sec >= target * 0.5


def test_process_event_video_recovers_when_chunk_file_corrupt(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    video_dir = dataset_root / "SportA" / "EventA"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "1.mp4"
    _make_test_video(video_path, duration_sec=10.0)

    out_root = tmp_path / "out"
    segment_path, _, _ = process_event_video(
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
    chunk_payload = json.loads((event_out / "chunk_captions.json").read_text(encoding="utf-8"))
    corrupt_chunk_path = Path(chunk_payload[0]["chunk_path"])
    corrupt_chunk_path.write_bytes(b"")  # invalid mp4 but file exists

    # Resume should rebuild the corrupt chunk video (ffprobe validation) and continue without crashing.
    process_event_video(
        event_video=EventVideo(sport="SportA", event="EventA", video_path=video_path),
        output_root=out_root,
        model=FakeCaptionModel(),
        rng=random.Random(1),
        language="zh",
        segment_min_sec=2.0,
        segment_max_sec=10.0,
        segment_fraction=0.8,
        chunk_sec=3.0,
        overwrite=False,
    )
    assert probe_video(segment_path).duration_sec > 0
    assert probe_video(corrupt_chunk_path).duration_sec > 0
