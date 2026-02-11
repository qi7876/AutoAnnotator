"""Utilities for reporting Video Captioner progress from on-disk outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .schema import DenseSegmentCaptionResponse


Status = Literal["completed", "partial", "not_started", "error"]


@dataclass(frozen=True)
class EventKey:
    sport: str
    event: str


@dataclass(frozen=True)
class EventProgress:
    key: EventKey
    status: Status
    duration_sec: float = 0.0
    span_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ProgressSummary:
    total_events: int
    completed_events: int
    partial_events: int
    not_started_events: int
    error_events: int
    completed_duration_sec: float
    completed_span_count: int
    by_sport_total: dict[str, int]
    by_sport_completed: dict[str, int]


def iter_dataset_events(dataset_root: Path) -> list[EventKey]:
    """Iterate data/caption_data/Dataset/{sport}/{event}/1.mp4 style datasets."""
    keys: list[EventKey] = []
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue
        for event_dir in sorted(sport_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            if (event_dir / "1.mp4").is_file():
                keys.append(EventKey(sport=sport_dir.name, event=event_dir.name))
    return keys


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_completed_stats(event_out: Path) -> tuple[float, int]:
    long_path = event_out / "long_caption.json"
    payload = _load_json(long_path)
    parsed = DenseSegmentCaptionResponse.model_validate(payload)
    duration_sec = float(parsed.info.total_frames) / float(parsed.info.fps)
    return duration_sec, len(parsed.spans)


def _read_partial_stats(event_out: Path) -> tuple[float, int]:
    chunk_path = event_out / "chunk_captions.json"
    payload = _load_json(chunk_path)
    if not isinstance(payload, list):
        raise ValueError("chunk_captions.json must be a JSON list")

    duration_sec = 0.0
    span_count = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        info = item.get("info")
        spans = item.get("spans")
        if isinstance(info, dict):
            fps = info.get("fps")
            total_frames = info.get("total_frames")
            try:
                fps_f = float(fps)
                frames_i = int(total_frames)
            except (TypeError, ValueError):
                fps_f = 0.0
                frames_i = 0
            if fps_f > 0 and frames_i > 0:
                duration_sec += frames_i / fps_f
        if isinstance(spans, list):
            span_count += sum(1 for s in spans if isinstance(s, dict) and isinstance(s.get("caption"), str))
    return duration_sec, span_count


def collect_progress(*, dataset_root: Path, output_root: Path) -> tuple[ProgressSummary, list[EventProgress]]:
    dataset_events = iter_dataset_events(dataset_root)

    by_sport_total: dict[str, int] = {}
    by_sport_completed: dict[str, int] = {}

    completed_events = 0
    partial_events = 0
    not_started_events = 0
    error_events = 0
    completed_duration_sec = 0.0
    completed_span_count = 0

    events: list[EventProgress] = []
    for key in dataset_events:
        by_sport_total[key.sport] = by_sport_total.get(key.sport, 0) + 1
        event_out = output_root / key.sport / key.event

        long_path = event_out / "long_caption.json"
        chunk_path = event_out / "chunk_captions.json"

        if long_path.is_file():
            try:
                duration_sec, span_count = _read_completed_stats(event_out)
            except Exception as exc:
                error_events += 1
                events.append(
                    EventProgress(key=key, status="error", error=f"failed to parse long_caption.json: {exc}")
                )
                continue

            completed_events += 1
            by_sport_completed[key.sport] = by_sport_completed.get(key.sport, 0) + 1
            completed_duration_sec += duration_sec
            completed_span_count += span_count
            events.append(
                EventProgress(key=key, status="completed", duration_sec=duration_sec, span_count=span_count)
            )
            continue

        if chunk_path.is_file():
            try:
                duration_sec, span_count = _read_partial_stats(event_out)
            except Exception as exc:
                error_events += 1
                events.append(
                    EventProgress(key=key, status="error", error=f"failed to parse chunk_captions.json: {exc}")
                )
                continue
            partial_events += 1
            events.append(EventProgress(key=key, status="partial", duration_sec=duration_sec, span_count=span_count))
            continue

        not_started_events += 1
        events.append(EventProgress(key=key, status="not_started"))

    summary = ProgressSummary(
        total_events=len(dataset_events),
        completed_events=completed_events,
        partial_events=partial_events,
        not_started_events=not_started_events,
        error_events=error_events,
        completed_duration_sec=completed_duration_sec,
        completed_span_count=completed_span_count,
        by_sport_total=by_sport_total,
        by_sport_completed=by_sport_completed,
    )
    return summary, events


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins, sec = divmod(int(round(seconds)), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m {sec}s"
    if mins:
        return f"{mins}m {sec}s"
    return f"{sec}s"
