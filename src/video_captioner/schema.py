"""Pydantic schemas for chunk-level and long-segment captions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, model_validator


class CaptionSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    caption: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_frame_order(self) -> "CaptionSpan":
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")
        return self


class ChunkCaptionResponse(BaseModel):
    """Model response for a ~1 minute chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_summary: str = Field(min_length=1)
    spans: list[CaptionSpan] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_non_overlapping_sorted(self) -> "ChunkCaptionResponse":
        prev_end = -1
        for span in self.spans:
            if span.start_frame <= prev_end:
                raise ValueError("spans must be sorted by start_frame and non-overlapping")
            prev_end = span.end_frame
        return self

    def validate_against_max_frame(self, max_frame: int) -> None:
        if max_frame < 0:
            raise ValueError("max_frame must be >= 0")
        for span in self.spans:
            if span.end_frame > max_frame:
                raise ValueError(f"span end_frame out of range: {span.end_frame} > {max_frame}")


@dataclass(frozen=True)
class SpanNormalizationInfo:
    """Metadata about how spans were normalized from raw model output."""

    mode: str  # "inclusive" or "exclusive"
    clamped: int
    shifted: int
    dropped: int


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid frame index")
    return int(value)


def _normalize_spans(
    spans: list[dict[str, Any]],
    *,
    max_frame: int,
    mode: str,
) -> tuple[list[dict[str, Any]], SpanNormalizationInfo]:
    clamped = 0
    shifted = 0
    dropped = 0

    normalized: list[dict[str, Any]] = []
    for span in spans:
        try:
            start = _coerce_int(span.get("start_frame"))
            end = _coerce_int(span.get("end_frame"))
        except Exception:
            dropped += 1
            continue

        if mode == "exclusive":
            # Convert [start, end) -> [start, end-1]
            end -= 1

        if start < 0:
            start = 0
            clamped += 1
        if start > max_frame:
            dropped += 1
            continue

        if end < 0:
            dropped += 1
            continue
        if end > max_frame:
            end = max_frame
            clamped += 1

        if end < start:
            dropped += 1
            continue

        normalized.append(
            {
                "start_frame": start,
                "end_frame": end,
                "caption": span["caption"],
            }
        )

    normalized.sort(key=lambda x: (int(x["start_frame"]), int(x["end_frame"])))

    cleaned: list[dict[str, Any]] = []
    prev_end = -1
    for span in normalized:
        start = int(span["start_frame"])
        end = int(span["end_frame"])
        if start <= prev_end:
            start = prev_end + 1
            shifted += 1
        if start > max_frame:
            dropped += 1
            continue
        if end < start:
            dropped += 1
            continue
        cleaned.append(
            {
                "start_frame": start,
                "end_frame": end,
                "caption": span["caption"],
            }
        )
        prev_end = end

    return cleaned, SpanNormalizationInfo(mode=mode, clamped=clamped, shifted=shifted, dropped=dropped)


def parse_chunk_caption_response(
    obj: Any,
    *,
    max_frame: int,
) -> tuple[ChunkCaptionResponse, SpanNormalizationInfo]:
    """
    Parse a model-produced chunk caption response and normalize spans into a strict, inclusive frame schema.

    This is resilient to common model mistakes:
    - using `end_frame` as an exclusive bound (half-open interval)
    - off-by-one `end_frame == max_frame + 1`
    - minor overlaps due to boundary ambiguity
    """
    if max_frame < 0:
        raise ValueError("max_frame must be >= 0")
    if isinstance(obj, str):
        # Sometimes models respond with a JSON string that itself contains JSON.
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            pass
    if isinstance(obj, list):
        # Some models mistakenly return a top-level array of spans.
        inferred_summary_parts: list[str] = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            caption = item.get("caption")
            if isinstance(caption, str) and caption.strip():
                inferred_summary_parts.append(caption.strip())
            if len(inferred_summary_parts) >= 2:
                break
        obj = {
            "chunk_summary": " ".join(inferred_summary_parts) if inferred_summary_parts else "Summary unavailable.",
            "spans": obj,
        }

    if not isinstance(obj, dict):
        raise ValueError("Chunk caption response must be a JSON object")

    # Accept common key aliases from non-strict outputs.
    chunk_summary = obj.get("chunk_summary")
    if chunk_summary is None:
        chunk_summary = obj.get("summary") or obj.get("chunkSummary") or obj.get("chunk_summary_text")
    if not isinstance(chunk_summary, str) or not chunk_summary.strip():
        raise ValueError("chunk_summary must be a non-empty string")

    raw_spans = obj.get("spans")
    if raw_spans is None:
        raw_spans = obj.get("segments") or obj.get("events") or obj.get("captions")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("spans must be a non-empty list")

    spans: list[dict[str, Any]] = []
    for item in raw_spans:
        if not isinstance(item, dict):
            continue
        caption = item.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            continue
        spans.append(
            {
                "start_frame": item.get("start_frame"),
                "end_frame": item.get("end_frame"),
                "caption": caption.strip(),
            }
        )

    if not spans:
        raise ValueError("spans must contain at least one valid item")

    # Heuristic: if the raw spans already satisfy strict non-overlap (inclusive) and
    # only exceed max_frame by <= 1, prefer inclusive clamping to avoid shrinking all spans.
    def _raw_sort_key(item: dict[str, Any]) -> tuple[int, int]:
        try:
            start_key = _coerce_int(item.get("start_frame"))
        except Exception:
            start_key = 0
        try:
            end_key = _coerce_int(item.get("end_frame"))
        except Exception:
            end_key = 0
        return start_key, end_key

    sorted_raw = sorted(spans, key=_raw_sort_key)
    raw_prev_end = -1
    raw_boundary_ties = 0
    raw_strict_ok = True
    raw_max_end: int | None = None
    for item in sorted_raw:
        try:
            start_i = _coerce_int(item.get("start_frame"))
            end_i = _coerce_int(item.get("end_frame"))
        except Exception:
            raw_strict_ok = False
            continue
        if start_i == raw_prev_end:
            raw_boundary_ties += 1
        if start_i <= raw_prev_end:
            raw_strict_ok = False
        raw_prev_end = end_i
        raw_max_end = end_i if raw_max_end is None else max(raw_max_end, end_i)

    if raw_max_end is None:
        raw_max_end = max_frame

    preferred_mode: str | None = None
    if raw_strict_ok and raw_boundary_ties == 0 and raw_max_end <= max_frame + 1:
        preferred_mode = "inclusive"
    elif raw_max_end == max_frame + 1 and raw_boundary_ties > 0:
        preferred_mode = "exclusive"

    candidates = []
    for mode in ("inclusive", "exclusive"):
        normalized, info = _normalize_spans(spans, max_frame=max_frame, mode=mode)
        # Prefer the heuristic mode when available; otherwise pick the best by minimal edits.
        score = (
            0 if preferred_mode is None or preferred_mode == mode else 1,
            info.dropped,
            info.shifted,
            info.clamped,
        )
        candidates.append((score, normalized, info))

    candidates.sort(key=lambda x: x[0])
    _, normalized_spans, info = candidates[0]
    resp = ChunkCaptionResponse.model_validate(
        {"chunk_summary": chunk_summary.strip(), "spans": normalized_spans}
    )
    resp.validate_against_max_frame(max_frame)
    return resp, info


class KeyMoment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_chunk_index: int = Field(ge=0)
    end_chunk_index: int = Field(ge=0)
    caption: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_order(self) -> "KeyMoment":
        if self.end_chunk_index < self.start_chunk_index:
            raise ValueError("end_chunk_index must be >= start_chunk_index")
        return self


class LongCaptionResponse(BaseModel):
    """Model response for the full extracted long segment."""

    model_config = ConfigDict(extra="forbid")

    long_caption: str = Field(min_length=1)
    key_moments: list[KeyMoment] = Field(min_length=1)

    @classmethod
    def from_model_json(cls, obj: Any) -> "LongCaptionResponse":
        # Convenience wrapper for symmetric API with ChunkCaptionResponse.
        return cls.model_validate(obj)


class ClipInfo(BaseModel):
    """AutoAnnotator-style frame mapping for a clip/segment."""

    model_config = ConfigDict(extra="forbid")

    original_starting_frame: int = Field(ge=0)
    total_frames: int = Field(gt=0)
    fps: float = Field(gt=0)


class DenseSpan(BaseModel):
    """A caption span mapped to original-video frame coordinates."""

    model_config = ConfigDict(extra="forbid")

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    caption: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_frame_order(self) -> "DenseSpan":
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be >= start_frame")
        return self


class DenseSegmentCaptionResponse(BaseModel):
    """Dense, play-by-play captions for the extracted long segment."""

    model_config = ConfigDict(extra="forbid")

    info: ClipInfo
    segment_summary: str = Field(min_length=1)
    spans: list[DenseSpan] = Field(min_length=1)
