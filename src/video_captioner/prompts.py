"""Prompt templates for chunk captioning and merge captioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # src/video_captioner/prompts.py -> repo root is ../../
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaptionPrompts:
    root_dir: Path = _repo_root() / "config" / "caption_prompts"

    def _read(self, filename: str) -> str:
        path = self.root_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8")

    def render_chunk_prompt(self, **vars: Any) -> str:
        template = self._read("chunk_caption.md")
        return template.format(**vars)

    def render_merge_prompt(self, **vars: Any) -> str:
        template = self._read("merge_caption.md")
        return template.format(**vars)

