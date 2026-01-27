"""End-to-end pipeline to generate captions for caption_data videos."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .ffmpeg_utils import (
    ChunkSpec,
    VideoProbe,
    keyframe_trim_copy,
    probe_video,
    select_random_segment,
    split_into_chunks,
)
from .model import CaptionModel, ChunkPromptContext
from .schema import ChunkCaptionResponse, DenseSegmentCaptionResponse


@dataclass(frozen=True)
class EventVideo:
    sport: str
    event: str
    video_path: Path


def resolve_dataset_root(path: Path) -> Path:
    """Accept either caption_data or caption_data/Dataset; return the dataset root."""
    if (path / "Dataset").is_dir():
        return path / "Dataset"
    return path


def iter_event_videos(dataset_root: Path, *, filename: str = "1.mp4") -> Iterator[EventVideo]:
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue
        for event_dir in sorted(sport_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            video_path = event_dir / filename
            if video_path.is_file():
                yield EventVideo(sport=sport_dir.name, event=event_dir.name, video_path=video_path)


@dataclass(frozen=True)
class ChunkCaptionRecord:
    chunk: ChunkSpec
    probe: VideoProbe
    response: ChunkCaptionResponse


@dataclass(frozen=True)
class LongCaptionRecord:
    response: DenseSegmentCaptionResponse


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_event_video(
    *,
    event_video: EventVideo,
    output_root: Path,
    model: CaptionModel,
    rng: random.Random,
    language: str = "zh",
    segment_min_sec: float = 5 * 60,
    segment_max_sec: float = 30 * 60,
    segment_fraction: float = 0.8,
    chunk_sec: float = 60.0,
    overwrite: bool = False,
) -> tuple[Path, list[ChunkCaptionRecord], LongCaptionRecord]:
    """
    Process a single {sport}/{event}/1.mp4 into:
    - one extracted long segment (keyframe copy)
    - many ~1 minute chunks (keyframe copy)
    - chunk captions + dense long-segment captions (no lossy merge)
    """
    out_dir = output_root / event_video.sport / event_video.event
    out_dir.mkdir(parents=True, exist_ok=True)

    segment_path = out_dir / "segment.mp4"
    chunks_dir = out_dir / "chunks"
    chunk_captions_path = out_dir / "chunk_captions.json"
    long_caption_path = out_dir / "long_caption.json"
    meta_path = out_dir / "run_meta.json"
    frame_fps = float(model.sampling_fps)

    if (
        not overwrite
        and chunk_captions_path.exists()
        and long_caption_path.exists()
        and segment_path.exists()
    ):
        chunk_payload = json.loads(chunk_captions_path.read_text(encoding="utf-8"))
        chunk_records: list[ChunkCaptionRecord] = []
        for item in chunk_payload:
            info = item.get("info") or {}
            fps_val = info.get("fps", item.get("fps"))
            total_frames_val = info.get("total_frames", item.get("total_frames"))
            chunk = ChunkSpec(
                index=int(item["chunk_index"]),
                start_sec=float(item["chunk_start_sec"]),
                duration_sec=float(item["chunk_duration_sec"]),
                path=Path(item["chunk_path"]),
            )
            probe = VideoProbe(
                duration_sec=float(item["chunk_duration_sec"]),
                start_time_sec=float(item.get("chunk_start_time_sec") or 0.0),
                fps=float(fps_val) if fps_val is not None else None,
                total_frames=int(total_frames_val) if total_frames_val is not None else None,
            )
            resp = ChunkCaptionResponse.model_validate(
                {"chunk_summary": item["chunk_summary"], "spans": item["spans"]}
            )
            if probe.total_frames is not None:
                resp.validate_against_max_frame(max(0, int(probe.total_frames) - 1))
            chunk_records.append(ChunkCaptionRecord(chunk=chunk, probe=probe, response=resp))

        dense_resp = DenseSegmentCaptionResponse.model_validate(
            json.loads(long_caption_path.read_text(encoding="utf-8"))
        )
        return segment_path, chunk_records, LongCaptionRecord(response=dense_resp)

    src_probe = probe_video(event_video.video_path)
    start_sec, dur_sec = select_random_segment(
        src_probe.duration_sec,
        rng=rng,
        min_duration_sec=segment_min_sec,
        max_duration_sec=segment_max_sec,
        fraction_of_total=segment_fraction,
    )

    keyframe_trim_copy(
        input_path=event_video.video_path,
        output_path=segment_path,
        start_sec=start_sec,
        duration_sec=dur_sec,
        overwrite=overwrite,
        preserve_timestamps=True,
    )

    segment_probe = probe_video(segment_path)
    segment_start_frame = int(round(segment_probe.start_time_sec * frame_fps))
    segment_total_frames = max(1, int(round(segment_probe.duration_sec * frame_fps)))

    chunks = split_into_chunks(
        input_path=segment_path,
        output_dir=chunks_dir,
        chunk_duration_sec=chunk_sec,
        overwrite=overwrite,
        preserve_timestamps=True,
    )

    chunk_records: list[ChunkCaptionRecord] = []
    previous_summary = ""
    for chunk in chunks:
        chunk_probe = probe_video(chunk.path)
        chunk_total_frames = max(1, int(round(chunk_probe.duration_sec * frame_fps)))
        duration_min = max(0.0, float(chunk_probe.duration_sec) / 60.0)
        min_spans = max(1, int(math.ceil(duration_min * 8)))
        max_spans = max(min_spans, int(math.ceil(duration_min * 18)))
        ctx = ChunkPromptContext(
            fps=frame_fps,
            total_frames=chunk_total_frames,
            max_frame=max(0, chunk_total_frames - 1),
        )
        resp = model.caption_chunk(
            video_path=chunk.path,
            ctx=ctx,
            language=language,
            previous_summary=previous_summary,
            min_spans=min_spans,
            max_spans=max_spans,
        )
        previous_summary = resp.chunk_summary
        chunk_records.append(ChunkCaptionRecord(chunk=chunk, probe=chunk_probe, response=resp))

    # Persist short chunk captions (with metadata) for later audits.
    chunk_payload = []
    for rec in chunk_records:
        chunk_start_frame = int(round(rec.probe.start_time_sec * frame_fps))
        chunk_total_frames = max(1, int(round(rec.probe.duration_sec * frame_fps)))
        chunk_payload.append(
            {
                "chunk_index": rec.chunk.index,
                "chunk_start_sec": rec.chunk.start_sec,
                "chunk_duration_sec": rec.probe.duration_sec,
                "chunk_start_time_sec": rec.probe.start_time_sec,
                "chunk_path": str(rec.chunk.path),
                "info": {
                    "original_starting_frame": chunk_start_frame,
                    "total_frames": chunk_total_frames,
                    "fps": frame_fps,
                },
                "chunk_summary": rec.response.chunk_summary,
                "spans": [span.model_dump() for span in rec.response.spans],
            }
        )
    _write_json(chunk_captions_path, chunk_payload)

    dense_spans = []
    summary_lines = []
    for item in chunk_payload:
        info = item["info"]
        base = int(info["original_starting_frame"])
        summary_lines.append(f'chunk_{int(item["chunk_index"]):03d}: {item["chunk_summary"]}')
        for span in item["spans"]:
            dense_spans.append(
                {
                    "start_frame": base + int(span["start_frame"]),
                    "end_frame": base + int(span["end_frame"]),
                    "caption": span["caption"],
                    "chunk_index": int(item["chunk_index"]),
                }
            )
    dense_spans.sort(key=lambda x: (x["start_frame"], x["end_frame"]))

    dense_payload = {
        "info": {
            "original_starting_frame": segment_start_frame,
            "total_frames": segment_total_frames,
            "fps": frame_fps,
        },
        "segment_summary": "\n".join(summary_lines) if summary_lines else "n/a",
        "spans": dense_spans,
    }
    dense_resp = DenseSegmentCaptionResponse.model_validate(dense_payload)
    long_record = LongCaptionRecord(response=dense_resp)
    _write_json(long_caption_path, dense_payload)

    _write_json(
        meta_path,
        {
            "source_video": str(event_video.video_path),
            "source_duration_sec": src_probe.duration_sec,
            "source_video_metadata": {
                "duration_sec": src_probe.duration_sec,
                "fps": frame_fps,
                "total_frames": max(1, int(round(src_probe.duration_sec * frame_fps))),
            },
            "segment_start_sec_requested": start_sec,
            "segment_duration_sec_target": dur_sec,
            "segment_start_time_sec": segment_probe.start_time_sec,
            "segment_duration_sec": segment_probe.duration_sec,
            "segment_info": {
                "original_starting_frame": segment_start_frame,
                "total_frames": segment_total_frames,
                "fps": frame_fps,
            },
            "segment_path": str(segment_path),
            "chunk_sec_target": chunk_sec,
            "num_chunks": len(chunk_records),
            "language": language,
        },
    )

    return segment_path, chunk_records, long_record


def process_many(
    *,
    dataset_root: Path,
    output_root: Path,
    model: CaptionModel,
    language: str,
    seed: int | None = None,
    sport: str | None = None,
    event: str | None = None,
    max_events: int | None = None,
    overwrite: bool = False,
    segment_min_sec: float = 5 * 60,
    segment_max_sec: float = 30 * 60,
    segment_fraction: float = 0.8,
    chunk_sec: float = 60.0,
) -> list[tuple[EventVideo, Path]]:
    rng = random.Random(seed)
    dataset_root = resolve_dataset_root(dataset_root)

    processed: list[tuple[EventVideo, Path]] = []
    for ev in iter_event_videos(dataset_root):
        if sport is not None and ev.sport != sport:
            continue
        if event is not None and ev.event != event:
            continue

        segment_path, _, _ = process_event_video(
            event_video=ev,
            output_root=output_root,
            model=model,
            rng=rng,
            language=language,
            segment_min_sec=segment_min_sec,
            segment_max_sec=segment_max_sec,
            segment_fraction=segment_fraction,
            chunk_sec=chunk_sec,
            overwrite=overwrite,
        )
        processed.append((ev, segment_path))
        if max_events is not None and len(processed) >= max_events:
            break
    return processed
