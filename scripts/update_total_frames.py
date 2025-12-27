#!/usr/bin/env python3
"""Update total_frames in clip metadata JSONs using ffprobe."""

import json
import subprocess
from pathlib import Path


def ffprobe_count_frames(video_path: Path, timeout_sec: int = 30) -> int:
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
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_sec
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timeout after {timeout_sec}s: {video_path}") from exc
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
    frame_counts: dict[int, int] = {}

    for json_path in clips_root.glob("*/*/clips/*.json"):
        scanned += 1
        try:
            print(f"scanning: {json_path}")
            if update_metadata_json(json_path, dataset_root):
                updated += 1
                print(f"updated: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_frames = data.get("info", {}).get("total_frames")
            if isinstance(total_frames, int):
                frame_counts[total_frames] = frame_counts.get(total_frames, 0) + 1
        except Exception as exc:
            print(f"failed: {json_path} ({exc})")

    print(f"scanned={scanned} updated={updated}")
    if frame_counts:
        total_items = sum(frame_counts.values())
        min_frames = min(frame_counts)
        max_frames = max(frame_counts)
        avg_frames = sum(k * v for k, v in frame_counts.items()) / total_items
        print("total_frames distribution:")
        print(f"min={min_frames} max={max_frames} avg={avg_frames:.2f}")
        for frames, count in sorted(frame_counts.items()):
            print(f"  {frames}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
