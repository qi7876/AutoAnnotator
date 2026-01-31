"""Tests for video_captioner.state."""

from __future__ import annotations

from pathlib import Path

from video_captioner.progress import EventKey
from video_captioner.state import VideoCaptionerState, load_state, save_state


def test_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = VideoCaptionerState(
        current=EventKey(sport="SportA", event="EventA"),
        processed={EventKey(sport="SportA", event="EventB"), EventKey(sport="SportB", event="EventC")},
    )
    save_state(path, state)

    loaded = load_state(path)
    assert loaded.current == EventKey(sport="SportA", event="EventA")
    assert loaded.processed == {
        EventKey(sport="SportA", event="EventB"),
        EventKey(sport="SportB", event="EventC"),
    }


def test_state_invalid_json_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    loaded = load_state(path)
    assert loaded.current is None
    assert loaded.processed == set()

