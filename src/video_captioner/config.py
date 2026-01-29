"""TOML-backed configuration for the video captioner CLI."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SegmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_sec: float = Field(default=5 * 60, gt=0)
    max_sec: float = Field(default=20 * 60, gt=0)
    fraction: float = Field(default=0.8, gt=0, lt=1)


class ChunkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sec: float = Field(default=60.0, gt=0)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO", min_length=1)
    file: Path | None = Field(default=None)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["gemini", "fake"] = "gemini"
    language: str = Field(default="en", min_length=1)
    seed: int | None = None
    sport: str | None = None
    event: str | None = None
    max_events: int | None = Field(default=None, ge=0)
    overwrite: bool = False
    progress: bool | None = None  # None => auto (TTY)

    @model_validator(mode="after")
    def _normalize_optionals(self) -> "RunConfig":
        if isinstance(self.sport, str) and not self.sport.strip():
            object.__setattr__(self, "sport", None)
        if isinstance(self.event, str) and not self.event.strip():
            object.__setattr__(self, "event", None)
        if self.max_events == 0:
            object.__setattr__(self, "max_events", None)
        return self


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=5, ge=1)
    wait_sec: float = Field(default=20.0, ge=0)
    jitter_sec: float = Field(default=2.0, ge=0)


class VideoCaptionerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_root: Path = Field(default=Path("caption_data"))
    output_root: Path = Field(default=Path("caption_outputs"))

    run: RunConfig = Field(default_factory=RunConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @classmethod
    def load(cls, path: Path) -> "VideoCaptionerConfig":
        if not path.is_file():
            raise FileNotFoundError(path)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config file must be a TOML table at the top level")
        return cls.model_validate(data)
