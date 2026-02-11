#!/usr/bin/env python3
"""Export completed Video Captioner outputs for manual checking."""

from __future__ import annotations

import argparse
from pathlib import Path

from video_captioner.checking_export import export_caption_checking
from video_captioner.config import VideoCaptionerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-captioner-export-checking")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/video_captioner_config.toml"),
        help="Path to TOML config (default: config/video_captioner_config.toml).",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("data/caption_checking"),
        help="Destination directory (default: data/caption_checking).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files under dest-root even if they exist.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar while exporting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg_path = Path(args.config)
    cfg = VideoCaptionerConfig.load(cfg_path)
    base_dir = cfg_path.resolve().parent

    dataset_root = cfg.dataset_root
    output_root = cfg.output_root
    dest_root = Path(args.dest_root)

    if not dataset_root.is_absolute():
        dataset_root = base_dir / dataset_root
    if not output_root.is_absolute():
        output_root = base_dir / output_root
    if not dest_root.is_absolute():
        dest_root = base_dir / dest_root

    report = export_caption_checking(
        dataset_root=dataset_root,
        output_root=output_root,
        dest_root=dest_root,
        overwrite=bool(args.overwrite),
        progress=True if args.progress else None,
    )

    print(
        f"Exported {len(report.completed_events)} completed event(s) into {report.dest_root} "
        f"(copied_files={report.copied_files}, skipped_files={report.skipped_files}, missing_files={report.missing_files})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
