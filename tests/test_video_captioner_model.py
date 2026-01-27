"""Tests for video_captioner.model."""

from __future__ import annotations

import json
from pathlib import Path

from video_captioner.model import ChunkPromptContext, GeminiCaptionModel, _parse_json_like


class _DummyGeminiClient:
    def __init__(self) -> None:
        self.upload_calls: list[Path] = []
        self.cleanup_calls: int = 0
        self.last_prompt: str | None = None
        self.last_timeout: int | None = None

    def upload_video(self, video_path: Path):
        self.upload_calls.append(video_path)
        return {"uri": "dummy://video"}

    def annotate_video(self, video_file, prompt: str, timeout=None):
        self.last_prompt = prompt
        self.last_timeout = timeout
        return {
            "chunk_summary": "summary",
            "spans": [
                {"start_frame": 0, "end_frame": 10, "caption": "a"},
                {"start_frame": 11, "end_frame": 20, "caption": "b"},
            ],
        }

    def cleanup_file(self, file_obj) -> None:
        self.cleanup_calls += 1


def test_gemini_caption_model_caption_chunk_renders_prompt_and_cleans_up(tmp_path: Path) -> None:
    dummy = _DummyGeminiClient()
    model = GeminiCaptionModel(gemini_client=dummy)

    video_path = tmp_path / "chunk.mp4"
    video_path.write_bytes(b"fake")  # never read by the dummy client

    resp = model.caption_chunk(
        video_path=video_path,
        ctx=ChunkPromptContext(fps=30.0, total_frames=21, max_frame=20),
        language="zh",
    )
    assert resp.chunk_summary == "summary"
    assert dummy.upload_calls == [video_path]
    assert dummy.cleanup_calls == 1
    assert dummy.last_prompt is not None and "0..20" in dummy.last_prompt


def test_gemini_caption_model_merge_long_caption_uses_prompt(monkeypatch) -> None:
    dummy = _DummyGeminiClient()
    model = GeminiCaptionModel(gemini_client=dummy)

    captured: dict[str, str] = {}

    def _fake_generate(prompt: str, *, timeout_sec):
        captured["prompt"] = prompt
        return {
            "long_caption": "merged",
            "key_moments": [{"start_chunk_index": 0, "end_chunk_index": 0, "caption": "m"}],
        }

    monkeypatch.setattr(model, "_generate_merge_json", _fake_generate)

    chunks_json = json.dumps([{"chunk_index": 0, "chunk_summary": "x"}], ensure_ascii=False)
    resp = model.merge_long_caption(chunks_json=chunks_json, language="zh")
    assert resp.long_caption == "merged"
    assert "chunk_index" in captured["prompt"]


def test_parse_json_like_handles_code_fences() -> None:
    raw = """```json
{"a": 1}
```"""
    assert _parse_json_like(raw) == {"a": 1}

