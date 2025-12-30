"""Persist editor state for resume."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EditorState:
    clip_index: int = 0
    frame_index: int = 1

    @classmethod
    def load(cls, path: Path) -> "EditorState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        return cls(
            clip_index=int(data.get("clip_index", 0)),
            frame_index=int(data.get("frame_index", 1)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"clip_index": self.clip_index, "frame_index": self.frame_index}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
