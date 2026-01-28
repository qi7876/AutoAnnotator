"""CLI smoke test for scripts/generate_captions.py (offline model)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from video_captioner.cli import main


def _make_test_video(path: Path, *, duration_sec: float, fps: int = 30) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={fps}",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=44100",
            "-t",
            f"{duration_sec:.3f}",
            "-c:v",
            "libopenh264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(fps),
            "-keyint_min",
            str(fps),
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cli_runs_end_to_end(tmp_path: Path) -> None:
    dataset_root = tmp_path / "caption_data" / "Dataset" / "SportA" / "EventA"
    dataset_root.mkdir(parents=True)
    video_path = dataset_root / "1.mp4"
    _make_test_video(video_path, duration_sec=6.0)

    out_root = tmp_path / "out"
    cfg_path = tmp_path / "video_captioner_config.toml"
    cfg_path.write_text(
        "\n".join(
            [
                'dataset_root = "caption_data"',
                'output_root = "out"',
                "",
                "[run]",
                'model = "fake"',
                'language = "en"',
                "max_events = 1",
                "seed = 0",
                "overwrite = true",
                "",
                "[segment]",
                "min_sec = 2",
                "max_sec = 10",
                "fraction = 0.8",
                "",
                "[chunk]",
                "sec = 3",
                "",
                "[logging]",
                'level = "INFO"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rc = main(
        [
            "--config",
            str(cfg_path),
        ]
    )
    assert rc == 0
    assert (out_root / "SportA" / "EventA" / "segment.mp4").is_file()
    assert (out_root / "SportA" / "EventA" / "chunk_captions.json").is_file()
    assert (out_root / "SportA" / "EventA" / "long_caption.json").is_file()
