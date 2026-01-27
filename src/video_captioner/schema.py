"""Pydantic schemas for chunk-level and long-segment captions."""

from __future__ import annotations

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

