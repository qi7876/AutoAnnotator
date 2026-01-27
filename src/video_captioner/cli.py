"""CLI entrypoint for the video captioner."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import FakeCaptionModel, GeminiCaptionModel
from .pipeline import process_many, resolve_dataset_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-captioner")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("caption_data"),
        help="Root containing Dataset/ (default: caption_data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("caption_outputs"),
        help="Where to write outputs (default: caption_outputs).",
    )
    parser.add_argument(
        "--model",
        choices=("gemini", "fake"),
        default="gemini",
        help="Caption model backend (default: gemini).",
    )
    parser.add_argument(
        "--language",
        default="zh",
        help="Output language hint for the model (default: zh).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    parser.add_argument("--sport", default=None, help="Only process a sport (optional).")
    parser.add_argument("--event", default=None, help="Only process an event (optional).")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after processing N events (optional).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs for an event.",
    )

    # Advanced knobs (useful for debugging; defaults satisfy the spec).
    parser.add_argument(
        "--segment-min-sec",
        type=float,
        default=5 * 60,
        help="Minimum extracted segment duration seconds (default: 300).",
    )
    parser.add_argument(
        "--segment-max-sec",
        type=float,
        default=30 * 60,
        help="Maximum extracted segment duration seconds (default: 1800).",
    )
    parser.add_argument(
        "--segment-fraction",
        type=float,
        default=0.8,
        help="Extracted segment target fraction of original duration (default: 0.8).",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=60.0,
        help="Chunk duration seconds (default: 60).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset_root = resolve_dataset_root(args.dataset_root)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    if args.model == "fake":
        model = FakeCaptionModel()
    else:
        model = GeminiCaptionModel()

    processed = process_many(
        dataset_root=dataset_root,
        output_root=args.output_root,
        model=model,
        language=str(args.language),
        seed=args.seed,
        sport=args.sport,
        event=args.event,
        max_events=args.max_events,
        overwrite=bool(args.overwrite),
        segment_min_sec=float(args.segment_min_sec),
        segment_max_sec=float(args.segment_max_sec),
        segment_fraction=float(args.segment_fraction),
        chunk_sec=float(args.chunk_sec),
    )

    print(f"Processed {len(processed)} event(s). Output root: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

