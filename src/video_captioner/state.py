"""Persistent run state for the video captioner.

We persist a small JSON file under the output root so that an interrupted run can
resume the last in-progress sport/event first, then continue with the remaining
events in a shuffled order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .progress import EventKey


STATE_VERSION = 1


@dataclass
class VideoCaptionerState:
    version: int = STATE_VERSION
    current: EventKey | None = None
    processed: set[EventKey] = field(default_factory=set)


def default_state_path(output_root: Path) -> Path:
    return output_root / "_state" / "video_captioner_state.json"


def _write_json_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _parse_event_key(obj: Any) -> EventKey | None:
    if not isinstance(obj, dict):
        return None
    sport = obj.get("sport")
    event = obj.get("event")
    if isinstance(sport, str) and isinstance(event, str) and sport and event:
        return EventKey(sport=sport, event=event)
    return None


def load_state(path: Path) -> VideoCaptionerState:
    if not path.is_file():
        return VideoCaptionerState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return VideoCaptionerState()
    if not isinstance(payload, dict):
        return VideoCaptionerState()

    state = VideoCaptionerState()

    current = _parse_event_key(payload.get("current"))
    if current is not None:
        state.current = current

    processed = payload.get("processed", [])
    if isinstance(processed, list):
        for item in processed:
            key = _parse_event_key(item)
            if key is not None:
                state.processed.add(key)

    return state


def save_state(path: Path, state: VideoCaptionerState) -> None:
    processed_sorted = sorted(state.processed, key=lambda k: (k.sport, k.event))
    payload = {
        "version": int(state.version),
        "current": None
        if state.current is None
        else {
            "sport": state.current.sport,
            "event": state.current.event,
        },
        "processed": [{"sport": k.sport, "event": k.event} for k in processed_sorted],
    }
    _write_json_atomic(path, payload)

