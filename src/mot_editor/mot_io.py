"""MOT format read/write helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class MotBox:
    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: float = -1.0
    x: float = -1.0
    y: float = -1.0
    z: float = -1.0

    def to_row(self) -> str:
        return (
            f"{self.frame},{self.track_id},"
            f"{self.left:.2f},{self.top:.2f},{self.width:.2f},{self.height:.2f},"
            f"{self.conf},{self.x},{self.y},{self.z}"
        )


class MotStore:
    """In-memory store for MOT boxes, indexed by frame."""

    def __init__(self) -> None:
        self._frames: Dict[int, List[MotBox]] = {}

    @property
    def frames(self) -> Dict[int, List[MotBox]]:
        return self._frames

    def get_frame(self, frame: int) -> List[MotBox]:
        return list(self._frames.get(frame, []))

    def set_frame(self, frame: int, boxes: List[MotBox]) -> None:
        self._frames[frame] = list(boxes)

    def update_box(self, frame: int, track_id: int, box: MotBox) -> None:
        boxes = self._frames.setdefault(frame, [])
        for i, existing in enumerate(boxes):
            if existing.track_id == track_id:
                boxes[i] = box
                return
        boxes.append(box)

    def max_frame(self) -> int:
        return max(self._frames.keys(), default=1)

    @classmethod
    def load(cls, path: Path) -> "MotStore":
        store = cls()
        if not path.exists():
            return store
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else -1.0
            x = float(parts[7]) if len(parts) > 7 else -1.0
            y = float(parts[8]) if len(parts) > 8 else -1.0
            z = float(parts[9]) if len(parts) > 9 else -1.0
            store.update_box(
                frame,
                track_id,
                MotBox(frame, track_id, left, top, width, height, conf, x, y, z),
            )
        return store

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame in sorted(self._frames.keys()):
            for box in sorted(self._frames[frame], key=lambda b: b.track_id):
                rows.append(box.to_row())
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        tmp_path.replace(path)
