"""Caption model interface and Gemini-backed implementation."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .prompts import CaptionPrompts
from .schema import ChunkCaptionResponse, LongCaptionResponse


@dataclass(frozen=True)
class ChunkPromptContext:
    fps: float
    total_frames: int
    max_frame: int


class CaptionModel(Protocol):
    def caption_chunk(
        self, *, video_path: Path, ctx: ChunkPromptContext, language: str
    ) -> ChunkCaptionResponse: ...

    def merge_long_caption(
        self, *, chunks_json: str, language: str
    ) -> LongCaptionResponse: ...


class GeminiCaptionModel:
    """
    Gemini-backed captioning model.

    This reuses AutoAnnotator's GeminiClient for:
    - uploading videos (incl. Vertex -> GCS upload)
    - generating JSON responses for video inputs
    """

    def __init__(
        self,
        *,
        gemini_client: Any = None,
        prompts: CaptionPrompts | None = None,
        video_timeout_sec: int | None = None,
        text_timeout_sec: int | None = None,
    ) -> None:
        if gemini_client is None:
            from auto_annotator.annotators.gemini_client import GeminiClient

            gemini_client = GeminiClient()
        self.gemini_client = gemini_client
        self.prompts = prompts or CaptionPrompts()
        self.video_timeout_sec = video_timeout_sec
        self.text_timeout_sec = text_timeout_sec

    def caption_chunk(
        self, *, video_path: Path, ctx: ChunkPromptContext, language: str
    ) -> ChunkCaptionResponse:
        prompt = self.prompts.render_chunk_prompt(
            language=language,
            fps=ctx.fps,
            total_frames=ctx.total_frames,
            max_frame=ctx.max_frame,
        )

        upload_path, cleanup_local = self._maybe_stage_for_vertex(video_path)
        video_file = self.gemini_client.upload_video(upload_path)
        try:
            raw = self.gemini_client.annotate_video(
                video_file, prompt, timeout=self.video_timeout_sec
            )
        finally:
            self.gemini_client.cleanup_file(video_file)
            cleanup_local()

        resp = ChunkCaptionResponse.model_validate(raw)
        resp.validate_against_max_frame(ctx.max_frame)
        return resp

    def merge_long_caption(
        self, *, chunks_json: str, language: str
    ) -> LongCaptionResponse:
        prompt = self.prompts.render_merge_prompt(
            language=language,
            chunks_json=chunks_json,
        )
        raw = self._generate_merge_json(prompt, timeout_sec=self.text_timeout_sec)
        return LongCaptionResponse.model_validate(raw)

    def _generate_merge_json(self, prompt: str, *, timeout_sec: int | None) -> Any:
        """
        Generate JSON from a text-only prompt using the same Gemini model.

        Kept as a separate method so tests can stub it without network calls.
        """
        from google.genai import types

        overrides: dict[str, Any] = {}
        if timeout_sec is not None:
            overrides["http_options"] = types.HttpOptions(timeout=int(timeout_sec * 1000))

        request_config = self.gemini_client._build_model_generation_config(
            overrides=overrides or None
        )
        response = self.gemini_client.model_client.models.generate_content(
            model=self.gemini_client.model_name,
            contents=[prompt],
            config=request_config,
        )

        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise ValueError("Gemini response did not contain text")

        # Keep parsing logic local to avoid relying on GeminiClient's private parser.
        parsed = _parse_json_like(text)
        return parsed

    def _maybe_stage_for_vertex(self, video_path: Path) -> tuple[Path, callable]:
        """
        Vertex backend uploads to GCS with object name derived from config.dataset_root.

        If `video_path` is outside config.dataset_root, GeminiClient would fall back to
        using only the basename as the object name (risking collisions across chunks).
        To avoid this, we stage the file under dataset_root with a hashed subdir.
        """
        backend = str(getattr(self.gemini_client, "model_backend", "")).lower()
        if backend not in {"vertexai", "vertex_ai", "vertex"}:
            return video_path, lambda: None

        dataset_root = Path(self.gemini_client.config.dataset_root)
        try:
            video_path.resolve().relative_to(dataset_root.resolve())
            return video_path, lambda: None
        except ValueError:
            pass

        key = str(video_path.resolve()).encode("utf-8")
        digest = __import__("hashlib").sha1(key).hexdigest()[:12]
        stage_dir = dataset_root / "_caption_upload_staging" / digest
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_path = stage_dir / video_path.name

        if stage_path.exists():
            stage_path.unlink()

        try:
            os.link(video_path, stage_path)
        except OSError:
            shutil.copy2(video_path, stage_path)

        def _cleanup() -> None:
            try:
                stage_path.unlink()
            except FileNotFoundError:
                return
            # Best-effort cleanup of empty staging dirs.
            try:
                stage_dir.rmdir()
                stage_dir.parent.rmdir()
            except OSError:
                return

        return stage_path, _cleanup


class FakeCaptionModel:
    """Deterministic caption model for unit tests and offline debugging."""

    def caption_chunk(
        self, *, video_path: Path, ctx: ChunkPromptContext, language: str
    ) -> ChunkCaptionResponse:
        mid = max(0, ctx.max_frame // 2)
        payload = {
            "chunk_summary": f"{language}: {video_path.name}",
            "spans": [
                {"start_frame": 0, "end_frame": mid, "caption": "first half"},
                {"start_frame": mid + 1, "end_frame": ctx.max_frame, "caption": "second half"},
            ],
        }
        resp = ChunkCaptionResponse.model_validate(payload)
        resp.validate_against_max_frame(ctx.max_frame)
        return resp

    def merge_long_caption(self, *, chunks_json: str, language: str) -> LongCaptionResponse:
        chunks = json.loads(chunks_json)
        payload = {
            "long_caption": f"{language}: merged {len(chunks)} chunks",
            "key_moments": [
                {"start_chunk_index": 0, "end_chunk_index": 0, "caption": "start"},
                {
                    "start_chunk_index": max(0, len(chunks) - 1),
                    "end_chunk_index": max(0, len(chunks) - 1),
                    "caption": "end",
                },
            ],
        }
        return LongCaptionResponse.model_validate(payload)


def _parse_json_like(text: str) -> Any:
    """Parse JSON possibly wrapped in Markdown fences."""
    s = text.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    return json.loads(s)
