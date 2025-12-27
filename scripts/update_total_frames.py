#!/usr/bin/env python3
"""Update total_frames in clip metadata JSONs using ffprobe."""

import json
import logging
import subprocess
from pathlib import Path


def ffprobe_count_frames(video_path: Path) -> int:
    """Return nb_read_frames from ffprobe."""
    cmd = [
        "ffprobe",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nw=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if line.startswith("nb_read_frames="):
            value = line.split("=", 1)[1].strip()
            return int(value)
    raise RuntimeError(f"nb_read_frames not found for {video_path}")


def update_metadata_json(json_path: Path, dataset_root: Path) -> bool:
    """Update total_frames; return True if updated."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    total_frames = info.get("total_frames")
    if total_frames is None:
        return False

    clip_id = data.get("id")
    origin = data.get("origin", {})
    sport = origin.get("sport")
    event = origin.get("event")
    if not (clip_id and sport and event):
        return False

    video_path = dataset_root / sport / event / "clips" / f"{clip_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    actual_frames = ffprobe_count_frames(video_path)
    if actual_frames == total_frames:
        return False

    info["total_frames"] = actual_frames
    data["info"] = info

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return True


def main() -> int:
    dataset_root = Path("data/Dataset")
    clips_root = dataset_root

    updated = 0
    scanned = 0

    for json_path in clips_root.glob("*/*/clips/*.json"):
        scanned += 1
        try:
            if update_metadata_json(json_path, dataset_root):
                updated += 1
                logger.info("updated: %s", json_path)
        except Exception as exc:
            logger.warning("failed: %s (%s)", json_path, exc)

    logger.info("scanned=%s updated=%s", scanned, updated)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
logger = logging.getLogger(__name__)
