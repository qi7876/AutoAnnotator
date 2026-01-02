#!/usr/bin/env python3
"""Re-encode clip videos with NVENC and refresh total_frames metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path

from update_total_frames import main as update_total_frames_main


def reencode_video(input_path: Path, timeout_sec: int = 600) -> None:
    temp_path = input_path.with_suffix(".reencoded.mp4")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "fps=10",
        "-fps_mode",
        "cfr",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p7",
        "-cq",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    subprocess.run(cmd, check=True, timeout=timeout_sec)
    temp_path.replace(input_path)


def main() -> int:
    dataset_root = Path("data/Dataset")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    clip_paths = sorted(dataset_root.glob("*/*/clips/*.mp4"))
    if not clip_paths:
        print("No clip videos found.")
        return 0

    total = len(clip_paths)
    for idx, clip_path in enumerate(clip_paths, start=1):
        print(f"[{idx}/{total}] re-encoding: {clip_path}")
        try:
            reencode_video(clip_path)
        except subprocess.TimeoutExpired:
            print(f"timeout: {clip_path}")
        except subprocess.CalledProcessError as exc:
            print(f"failed: {clip_path} ({exc})")

    print("Updating total_frames metadata...")
    update_total_frames_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
