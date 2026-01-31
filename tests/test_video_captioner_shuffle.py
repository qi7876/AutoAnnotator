"""Tests for event ordering utilities in video_captioner.pipeline."""

from __future__ import annotations

import random
from pathlib import Path

from video_captioner.pipeline import EventVideo, shuffle_events_even_across_sports


def test_shuffle_events_even_across_sports_interleaves_sports() -> None:
    events = [
        EventVideo(sport="SportA", event="E1", video_path=Path("a1.mp4")),
        EventVideo(sport="SportA", event="E2", video_path=Path("a2.mp4")),
        EventVideo(sport="SportB", event="F1", video_path=Path("b1.mp4")),
        EventVideo(sport="SportB", event="F2", video_path=Path("b2.mp4")),
    ]

    ordered = shuffle_events_even_across_sports(events, rng=random.Random(0))
    assert {f"{e.sport}/{e.event}" for e in ordered} == {f"{e.sport}/{e.event}" for e in events}

    # With balanced sports counts, we should not see the same sport twice in a row.
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        assert prev.sport != cur.sport

