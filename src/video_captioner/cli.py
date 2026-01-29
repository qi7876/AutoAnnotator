"""CLI entrypoint for the video captioner."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import VideoCaptionerConfig
from .logging_utils import configure_logging
from .model import FakeCaptionModel, GeminiCaptionModel
from .pipeline import process_many, resolve_dataset_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-captioner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("video_captioner_config.toml"),
        help="Path to TOML config (default: video_captioner_config.toml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite for this run (overrides config.run.overwrite).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = Path(args.config)
    cfg = VideoCaptionerConfig.load(config_path)
    base_dir = config_path.resolve().parent

    dataset_root = cfg.dataset_root
    if not dataset_root.is_absolute():
        dataset_root = base_dir / dataset_root

    output_root = cfg.output_root
    if not output_root.is_absolute():
        output_root = base_dir / output_root

    dataset_root = resolve_dataset_root(dataset_root)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    log_file = cfg.logging.file
    if log_file is None:
        log_file = output_root / "_logs" / "video_captioner.log"
    elif not log_file.is_absolute():
        log_file = output_root / log_file
    configure_logging(log_file=Path(log_file), level=str(cfg.logging.level))

    if cfg.run.model == "fake":
        model = FakeCaptionModel()
    else:
        model = GeminiCaptionModel(
            retry_max_attempts=int(cfg.retry.max_attempts),
            retry_wait_sec=float(cfg.retry.wait_sec),
            retry_jitter_sec=float(cfg.retry.jitter_sec),
        )

    processed = process_many(
        dataset_root=dataset_root,
        output_root=output_root,
        model=model,
        language=str(cfg.run.language),
        seed=cfg.run.seed,
        sport=cfg.run.sport,
        event=cfg.run.event,
        max_events=cfg.run.max_events,
        overwrite=bool(args.overwrite or cfg.run.overwrite),
        segment_min_sec=float(cfg.segment.min_sec),
        segment_max_sec=float(cfg.segment.max_sec),
        segment_fraction=float(cfg.segment.fraction),
        chunk_sec=float(cfg.chunk.sec),
        progress=cfg.run.progress,
    )

    print(f"Processed {len(processed)} event(s). Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
