"""Tests for video_captioner.schema and prompt rendering."""

from __future__ import annotations

import json

import pytest

from video_captioner.prompts import CaptionPrompts
from video_captioner.schema import (
    ChunkCaptionResponse,
    DenseSegmentCaptionResponse,
    LongCaptionResponse,
    parse_chunk_caption_response,
)


def test_chunk_caption_schema_rejects_overlapping_spans() -> None:
    with pytest.raises(ValueError):
        ChunkCaptionResponse.model_validate(
            {
                "chunk_summary": "summary",
                "spans": [
                    {"start_frame": 0, "end_frame": 10, "caption": "a"},
                    {"start_frame": 10, "end_frame": 20, "caption": "b"},
                ],
            }
        )


def test_chunk_caption_schema_validates_max_frame() -> None:
    resp = ChunkCaptionResponse.model_validate(
        {
            "chunk_summary": "summary",
            "spans": [
                {"start_frame": 0, "end_frame": 10, "caption": "a"},
                {"start_frame": 11, "end_frame": 20, "caption": "b"},
            ],
        }
    )
    resp.validate_against_max_frame(20)
    with pytest.raises(ValueError):
        resp.validate_against_max_frame(19)


def test_parse_chunk_caption_response_accepts_exclusive_end_frame_convention() -> None:
    # Simulate common model output that uses [start, end) half-open ranges and
    # produces boundary ties (next.start_frame == prev.end_frame).
    raw = {
        "chunk_summary": "summary",
        "spans": [
            {"start_frame": 0, "end_frame": 10, "caption": "a"},
            {"start_frame": 10, "end_frame": 20, "caption": "b"},
            {"start_frame": 20, "end_frame": 21, "caption": "c"},
        ],
    }
    resp, info = parse_chunk_caption_response(raw, max_frame=20)
    assert info.mode == "exclusive"
    assert [(s.start_frame, s.end_frame) for s in resp.spans] == [(0, 9), (10, 19), (20, 20)]


def test_parse_chunk_caption_response_accepts_top_level_span_list() -> None:
    raw = [
        {"start_frame": 0, "end_frame": 10, "caption": "a"},
        {"start_frame": 11, "end_frame": 20, "caption": "b"},
    ]
    resp, info = parse_chunk_caption_response(raw, max_frame=20)
    assert info.mode in {"inclusive", "exclusive"}
    assert resp.chunk_summary
    assert [(s.start_frame, s.end_frame) for s in resp.spans] == [(0, 10), (11, 20)]


def test_parse_chunk_caption_response_prefers_inclusive_when_only_last_is_off_by_one() -> None:
    raw = {
        "chunk_summary": "summary",
        "spans": [
            {"start_frame": 0, "end_frame": 10, "caption": "a"},
            {"start_frame": 11, "end_frame": 21, "caption": "b"},
        ],
    }
    resp, info = parse_chunk_caption_response(raw, max_frame=20)
    assert info.mode == "inclusive"
    assert [(s.start_frame, s.end_frame) for s in resp.spans] == [(0, 10), (11, 20)]


def test_long_caption_schema() -> None:
    resp = LongCaptionResponse.from_model_json(
        {
            "long_caption": "all",
            "key_moments": [{"start_chunk_index": 0, "end_chunk_index": 1, "caption": "m"}],
        }
    )
    assert resp.long_caption == "all"
    assert len(resp.key_moments) == 1


def test_prompts_render() -> None:
    prompts = CaptionPrompts()
    chunk_prompt = prompts.render_chunk_prompt(
        language="en",
        fps=30,
        total_frames=60,
        max_frame=59,
        previous_summary="上一段总结",
        min_spans=8,
        max_spans=18,
    )
    assert "Language: en" in chunk_prompt
    assert "0..59" in chunk_prompt
    assert "Previous chunk summary" in chunk_prompt
    assert "between 8 and 18" in chunk_prompt

    merge_prompt = prompts.render_merge_prompt(
        language="en",
        chunks_json=json.dumps([{"chunk_index": 0, "chunk_summary": "x"}]),
    )
    assert "Language: en" in merge_prompt
    assert "Input chunk captions" in merge_prompt


def test_dense_segment_caption_schema() -> None:
    resp = DenseSegmentCaptionResponse.model_validate(
        {
            "info": {"original_starting_frame": 100, "total_frames": 50, "fps": 10},
            "segment_summary": "s",
            "spans": [
                {"start_frame": 100, "end_frame": 110, "caption": "a", "chunk_index": 0}
            ],
        }
    )
    assert resp.info.original_starting_frame == 100
