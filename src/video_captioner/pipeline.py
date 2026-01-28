"""End-to-end pipeline to generate captions for caption_data videos."""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from loguru import logger

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from .ffmpeg_utils import (
    ChunkSpec,
    VideoProbe,
    keyframe_trim_copy,
    probe_video,
    select_random_segment,
    split_into_chunks,
)
from .model import CaptionModel, ChunkPromptContext
from .schema import ChunkCaptionResponse, DenseSegmentCaptionResponse, parse_chunk_caption_response


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
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _progress_enabled(enabled: bool | None) -> bool:
    if enabled is None:
        return sys.stderr.isatty()
    return bool(enabled)


def _load_json_list(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON (will treat as empty and continue): {path}")
        return []
    if not isinstance(payload, list):
        logger.warning(f"Expected JSON list but got {type(payload).__name__}: {path}")
        return []
    return [item for item in payload if isinstance(item, dict)]


def _load_json_dict(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON (will treat as empty and continue): {path}")
        return {}
    if not isinstance(payload, dict):
        logger.warning(f"Expected JSON object but got {type(payload).__name__}: {path}")
        return {}
    return payload


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _segment_duration_too_short(*, actual_sec: float, target_sec: float) -> bool:
    if target_sec <= 0:
        return False
    ratio = actual_sec / target_sec
    min_ratio = 0.8 if target_sec >= 60.0 else 0.5
    return ratio < min_ratio


def _build_chunk_payload_item(
    *,
    chunk: ChunkSpec,
    probe: VideoProbe,
    resp: ChunkCaptionResponse,
    frame_fps: float,
) -> dict:
    chunk_start_frame = int(round(probe.start_time_sec * frame_fps))
    chunk_total_frames = max(1, int(round(probe.duration_sec * frame_fps)))
    return {
        "chunk_index": chunk.index,
        "chunk_start_sec": chunk.start_sec,
        "chunk_duration_sec": probe.duration_sec,
        "chunk_start_time_sec": probe.start_time_sec,
        "chunk_path": str(chunk.path),
        "info": {
            "original_starting_frame": chunk_start_frame,
            "total_frames": chunk_total_frames,
            "fps": frame_fps,
        },
        "chunk_summary": resp.chunk_summary,
        "spans": [span.model_dump() for span in resp.spans],
    }


def process_event_video(
    *,
    event_video: EventVideo,
    output_root: Path,
    model: CaptionModel,
    rng: random.Random,
    language: str = "en",
    segment_min_sec: float = 5 * 60,
    segment_max_sec: float = 30 * 60,
    segment_fraction: float = 0.8,
    chunk_sec: float = 60.0,
    overwrite: bool = False,
    progress: bool | None = None,
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

    progress_on = _progress_enabled(progress)

    src_probe = probe_video(event_video.video_path)

    meta_existing = _load_json_dict(meta_path) if meta_path.exists() else {}
    recorded_start_sec = _coerce_float(meta_existing.get("segment_start_sec_requested"))
    recorded_dur_sec = _coerce_float(meta_existing.get("segment_duration_sec_target"))

    start_sec: float | None = None
    dur_sec: float | None = None

    segment_probe: VideoProbe | None = None
    segment_needs_recut = False
    if segment_path.exists() and not overwrite:
        try:
            segment_probe = probe_video(segment_path)
        except Exception as exc:
            logger.warning(f"Segment exists but is not playable; will re-extract: {exc}")
            segment_needs_recut = True
        else:
            if recorded_dur_sec is not None and _segment_duration_too_short(
                actual_sec=float(segment_probe.duration_sec),
                target_sec=float(recorded_dur_sec),
            ):
                logger.warning(
                    f"Segment duration is too short ({segment_probe.duration_sec:.2f}s) "
                    f"vs expected target ({recorded_dur_sec:.2f}s); will re-extract."
                )
                segment_needs_recut = True

    # Ensure we have a valid segment file before proceeding (and before any early-return).
    if overwrite or segment_needs_recut or not segment_path.exists():
        use_recorded = (
            not overwrite and recorded_start_sec is not None and recorded_dur_sec is not None
        )
        if use_recorded:
            start_sec = float(recorded_start_sec)
            dur_sec = float(recorded_dur_sec)
        else:
            start_sec, dur_sec = select_random_segment(
                src_probe.duration_sec,
                rng=rng,
                min_duration_sec=segment_min_sec,
                max_duration_sec=segment_max_sec,
                fraction_of_total=segment_fraction,
            )

        _write_json(
            meta_path,
            {
                **meta_existing,
                "source_video": str(event_video.video_path),
                "source_duration_sec": src_probe.duration_sec,
                "segment_start_sec_requested": start_sec,
                "segment_duration_sec_target": dur_sec,
                "chunk_sec_target": chunk_sec,
                "language": language,
                "status": "segment_requested",
            },
        )

        logger.info(
            f"Extracting segment (requested start={start_sec:.2f}s, dur={dur_sec:.2f}s) -> {segment_path}"
        )

        # If we cannot reuse the recorded request, this segment will not match existing outputs.
        if not use_recorded and not overwrite:
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir, ignore_errors=True)
            for path in (chunk_captions_path, long_caption_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        keyframe_trim_copy(
            input_path=event_video.video_path,
            output_path=segment_path,
            start_sec=float(start_sec),
            duration_sec=float(dur_sec),
            overwrite=True if (overwrite or segment_needs_recut) else overwrite,
            preserve_timestamps=True,
        )
        segment_probe = probe_video(segment_path)
    else:
        assert segment_probe is not None
        logger.info(f"Resuming with existing segment: {segment_path}")

    # Fast path: if all outputs exist and media files are valid, load and return.
    if not overwrite and chunk_captions_path.exists() and long_caption_path.exists():
        try:
            probe_video(segment_path)
            chunk_payload = _load_json_list(chunk_captions_path)
            chunk_paths = []
            for item in chunk_payload:
                p = Path(str(item.get("chunk_path") or ""))
                if not p.is_file():
                    raise FileNotFoundError(p)
                probe_video(p)
                chunk_paths.append(p)
            if not chunk_payload:
                raise ValueError("chunk_captions.json is empty")

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
                max_frame = max(0, int(probe.total_frames or 1) - 1)
                resp, _ = parse_chunk_caption_response(
                    {"chunk_summary": item.get("chunk_summary"), "spans": item.get("spans")},
                    max_frame=max_frame,
                )
                chunk_records.append(ChunkCaptionRecord(chunk=chunk, probe=probe, response=resp))

            dense_resp = DenseSegmentCaptionResponse.model_validate(
                json.loads(long_caption_path.read_text(encoding="utf-8"))
            )
            return segment_path, chunk_records, LongCaptionRecord(response=dense_resp)
        except Exception as exc:
            logger.warning(f"Existing outputs look incomplete/corrupt; will resume instead: {exc}")

    segment_start_frame = int(round(segment_probe.start_time_sec * frame_fps))
    segment_total_frames = max(1, int(round(segment_probe.duration_sec * frame_fps)))

    chunks = split_into_chunks(
        input_path=segment_path,
        output_dir=chunks_dir,
        chunk_duration_sec=chunk_sec,
        overwrite=overwrite,
        preserve_timestamps=True,
    )

    existing_by_index: dict[int, dict] = {}
    if chunk_captions_path.exists() and not overwrite:
        for item in _load_json_list(chunk_captions_path):
            idx = item.get("chunk_index")
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            existing_by_index[idx_int] = item
        if existing_by_index:
            logger.info(f"Found {len(existing_by_index)} existing chunk caption(s); will resume.")

    chunk_records: list[ChunkCaptionRecord] = []
    previous_summary = ""
    chunks_iter: Iterable[ChunkSpec] = chunks
    if tqdm is not None and progress_on:
        chunks_iter = tqdm(
            chunks,
            desc=f"{event_video.sport}/{event_video.event}",
            unit="chunk",
            leave=False,
        )

    wrote_chunk_captions = False
    for chunk in chunks_iter:
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
        existing_item = existing_by_index.get(chunk.index) if not overwrite else None
        if existing_item is not None:
            resp, info = parse_chunk_caption_response(
                {"chunk_summary": existing_item.get("chunk_summary"), "spans": existing_item.get("spans")},
                max_frame=ctx.max_frame,
            )
            if info.mode != "inclusive" or info.clamped or info.shifted or info.dropped:
                logger.warning(
                    f"Normalized existing chunk_{chunk.index:03d} spans "
                    f"(mode={info.mode}, clamped={info.clamped}, shifted={info.shifted}, dropped={info.dropped})."
                )
        else:
            logger.info(
                f"Captioning chunk_{chunk.index:03d} ({chunk_probe.duration_sec:.2f}s, max_frame={ctx.max_frame})"
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

        item = _build_chunk_payload_item(chunk=chunk, probe=chunk_probe, resp=resp, frame_fps=frame_fps)
        if existing_item != item:
            existing_by_index[chunk.index] = item
            wrote_chunk_captions = True
            chunk_payload = [existing_by_index[i] for i in sorted(existing_by_index)]
            _write_json(chunk_captions_path, chunk_payload)

    # Persist final short chunk captions (with metadata) for later audits.
    chunk_payload = [_build_chunk_payload_item(chunk=rec.chunk, probe=rec.probe, resp=rec.response, frame_fps=frame_fps) for rec in chunk_records]
    if overwrite or wrote_chunk_captions or not chunk_captions_path.exists():
        _write_json(chunk_captions_path, chunk_payload)

    dense_spans = []
    summary_lines = []
    segment_max_frame = segment_start_frame + segment_total_frames - 1
    for item in chunk_payload:
        info = item["info"]
        base = int(info["original_starting_frame"])
        summary_lines.append(f'chunk_{int(item["chunk_index"]):03d}: {item["chunk_summary"]}')
        for span in item["spans"]:
            start = base + int(span["start_frame"])
            end = base + int(span["end_frame"])
            if start < segment_start_frame:
                start = segment_start_frame
            if end > segment_max_frame:
                end = segment_max_frame
            if end < start:
                continue
            dense_spans.append(
                {
                    "start_frame": start,
                    "end_frame": end,
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
    if overwrite or wrote_chunk_captions or not long_caption_path.exists():
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
    progress: bool | None = None,
) -> list[tuple[EventVideo, Path]]:
    rng = random.Random(seed)
    dataset_root = resolve_dataset_root(dataset_root)

    processed: list[tuple[EventVideo, Path]] = []
    progress_on = _progress_enabled(progress)

    events_iter: Iterable[EventVideo] = iter_event_videos(dataset_root)
    if tqdm is not None and progress_on:
        events_iter = tqdm(events_iter, desc="Events", unit="event")

    for ev in events_iter:
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
            progress=progress,
        )
        processed.append((ev, segment_path))
        if max_events is not None and len(processed) >= max_events:
            break
    return processed
