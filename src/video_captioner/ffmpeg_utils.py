"""ffprobe/ffmpeg helpers for keyframe-only trimming and chunking."""

from __future__ import annotations

import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FfmpegCommandError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stderr: str):
        super().__init__(
            f"ffmpeg/ffprobe command failed (code={returncode}): {' '.join(cmd)}\n{stderr}"
        )
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class VideoProbe:
    duration_sec: float
    start_time_sec: float = 0.0
    fps: float | None = None
    total_frames: int | None = None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FfmpegCommandError(cmd=cmd, returncode=proc.returncode, stderr=proc.stderr)
    return proc


def _parse_fraction(value: str) -> float | None:
    value = value.strip()
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" not in value:
        try:
            return float(value)
        except ValueError:
            return None
    num_str, den_str = value.split("/", 1)
    try:
        num = float(num_str)
        den = float(den_str)
    except ValueError:
        return None
    if den == 0:
        return None
    return num / den


def probe_video(video_path: Path) -> VideoProbe:
    """Return duration (seconds) and best-effort fps/frames for a media file."""
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
    )
    data = json.loads(proc.stdout)

    duration = None
    start_time = 0.0
    fmt = data.get("format") or {}
    start_time_str = fmt.get("start_time")
    if start_time_str not in (None, "N/A"):
        try:
            start_time = float(start_time_str)
        except (TypeError, ValueError):
            start_time = 0.0
    duration_str = fmt.get("duration")
    if duration_str is not None:
        try:
            duration = float(duration_str)
        except (TypeError, ValueError):
            duration = None
    if duration is None or duration <= 0:
        raise ValueError(f"Failed to probe duration for: {video_path}")

    fps: float | None = None
    total_frames: int | None = None

    # Prefer the first video stream.
    for stream in data.get("streams") or []:
        if stream.get("codec_type") != "video":
            continue
        fps = _parse_fraction(stream.get("avg_frame_rate", "")) or _parse_fraction(
            stream.get("r_frame_rate", "")
        )
        nb_frames = stream.get("nb_frames")
        if nb_frames not in (None, "N/A"):
            try:
                total_frames = int(nb_frames)
            except (TypeError, ValueError):
                total_frames = None
        break

    if total_frames is None and fps is not None:
        total_frames = int(round(duration * fps))

    return VideoProbe(
        duration_sec=duration,
        start_time_sec=start_time,
        fps=fps,
        total_frames=total_frames,
    )


def select_random_segment(
    total_duration_sec: float,
    *,
    rng: random.Random,
    min_duration_sec: float = 5 * 60,
    max_duration_sec: float = 30 * 60,
    fraction_of_total: float = 0.8,
) -> tuple[float, float]:
    """
    Select a random segment (start_sec, duration_sec) from a video.

    Rules:
    - If total < min_duration: use the full video (start=0).
    - Otherwise choose duration in [min_duration, max_duration], increasing with total.
    - Start position is uniform random within [0, total-duration].
    """
    if total_duration_sec <= 0:
        raise ValueError("total_duration_sec must be positive")
    if min_duration_sec <= 0 or max_duration_sec <= 0:
        raise ValueError("min/max duration must be positive")
    if min_duration_sec > max_duration_sec:
        raise ValueError("min_duration_sec must be <= max_duration_sec")
    if not (0 < fraction_of_total < 1):
        raise ValueError("fraction_of_total must be in (0, 1)")

    if total_duration_sec <= min_duration_sec:
        return 0.0, float(total_duration_sec)

    desired = total_duration_sec * fraction_of_total
    desired = max(min_duration_sec, desired)
    desired = min(max_duration_sec, desired)
    desired = min(total_duration_sec, desired)

    max_start = max(0.0, total_duration_sec - desired)
    start = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
    return float(start), float(desired)


def keyframe_trim_copy(
    *,
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    overwrite: bool = False,
    preserve_timestamps: bool = False,
) -> bool:
    """
    Trim by keyframe-aligned seeking and stream-copy (no re-encode).

    Uses `-ss` before `-i` for keyframe seek. The actual start may be earlier than
    `start_sec` depending on GOP structure.
    """
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if start_sec < 0:
        raise ValueError("start_sec must be >= 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        should_rebuild = False
        try:
            if output_path.stat().st_size <= 0:
                should_rebuild = True
            else:
                # ffprobe validation: non-empty doesn't mean playable.
                probe_video(output_path)
        except Exception:
            should_rebuild = True

        if not should_rebuild:
            return False
        overwrite = True

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if overwrite:
        cmd.append("-y")
    cmd += ["-ss", f"{start_sec:.3f}", "-i", str(input_path)]
    if preserve_timestamps:
        input_start_time = probe_video(input_path).start_time_sec
        end_sec = input_start_time + start_sec + duration_sec
        cmd += [
            "-to",
            f"{end_sec:.3f}",
            "-map",
            "0",
            "-c",
            "copy",
            "-copyts",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    else:
        cmd += [
            "-t",
            f"{duration_sec:.3f}",
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    _run(cmd)
    return True


@dataclass(frozen=True)
class ChunkSpec:
    index: int
    start_sec: float
    duration_sec: float
    path: Path


def split_into_chunks(
    *,
    input_path: Path,
    output_dir: Path,
    chunk_duration_sec: float = 60.0,
    overwrite: bool = False,
    preserve_timestamps: bool = False,
) -> list[ChunkSpec]:
    """Split a media file into ~fixed-duration chunks using keyframe stream copy."""
    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be positive")

    probe = probe_video(input_path)
    total = probe.duration_sec
    if total <= 0:
        raise ValueError(f"Invalid duration for {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix or ".mp4"

    num_chunks = int(math.ceil(total / chunk_duration_sec))
    chunks: list[ChunkSpec] = []
    for idx in range(num_chunks):
        start = idx * chunk_duration_sec
        remaining = max(0.0, total - start)
        if remaining <= 0:
            break
        dur = min(chunk_duration_sec, remaining)
        out_path = output_dir / f"chunk_{idx:03d}{suffix}"
        keyframe_trim_copy(
            input_path=input_path,
            output_path=out_path,
            start_sec=start,
            duration_sec=dur,
            overwrite=overwrite,
            preserve_timestamps=preserve_timestamps,
        )
        chunks.append(ChunkSpec(index=idx, start_sec=start, duration_sec=dur, path=out_path))

    if not chunks:
        raise RuntimeError(f"No chunks produced for {input_path}")
    return chunks
