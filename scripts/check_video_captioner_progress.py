#!/usr/bin/env python3
"""Check Video Captioner progress from caption_outputs on disk.

This script scans:
- Dataset: caption_data/Dataset/{sport}/{event}/1.mp4
- Outputs: caption_outputs/{sport}/{event}/long_caption.json (completed)

It reports how many sport/event have finished, plus total annotated duration and caption span count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_captioner.config import VideoCaptionerConfig
from video_captioner.pipeline import resolve_dataset_root
from video_captioner.progress import collect_progress, format_duration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-captioner-progress")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("video_captioner_config.toml"),
        help="Path to TOML config (default: video_captioner_config.toml).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset root (caption_data or caption_data/Dataset).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root (default from config).",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-event status lines (can be long).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable JSON summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg_path = Path(args.config)
    cfg = VideoCaptionerConfig.load(cfg_path)
    base_dir = cfg_path.resolve().parent

    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else cfg.dataset_root
    output_root = Path(args.output_root) if args.output_root is not None else cfg.output_root

    if not dataset_root.is_absolute():
        dataset_root = base_dir / dataset_root
    if not output_root.is_absolute():
        output_root = base_dir / output_root

    dataset_root = resolve_dataset_root(dataset_root)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    if not output_root.exists():
        print(f"[warn] Output root does not exist yet: {output_root}")

    summary, events = collect_progress(dataset_root=dataset_root, output_root=output_root)

    pct = (100.0 * summary.completed_events / summary.total_events) if summary.total_events else 0.0

    print("Video Captioner progress")
    print(f"  Dataset events: {summary.total_events}")
    print(f"  Completed: {summary.completed_events} ({pct:.1f}%)")
    if summary.partial_events:
        print(f"  Partial: {summary.partial_events}")
    if summary.not_started_events:
        print(f"  Not started: {summary.not_started_events}")
    if summary.error_events:
        print(f"  Errors: {summary.error_events}")
    print(f"  Completed duration: {format_duration(summary.completed_duration_sec)}")
    print(f"  Completed caption spans: {summary.completed_span_count}")

    if summary.by_sport_total:
        print("")
        print("By sport (completed/total)")
        for sport in sorted(summary.by_sport_total):
            done = summary.by_sport_completed.get(sport, 0)
            total = summary.by_sport_total[sport]
            print(f"  {sport}: {done}/{total}")

    if args.details:
        print("")
        print("Event details")
        for ev in events:
            key = f"{ev.key.sport}/{ev.key.event}"
            if ev.status == "completed":
                print(f"  [done] {key}  {format_duration(ev.duration_sec)}  spans={ev.span_count}")
            elif ev.status == "partial":
                print(f"  [partial] {key}  ~{format_duration(ev.duration_sec)}  spans~={ev.span_count}")
            elif ev.status == "error":
                print(f"  [error] {key}  {ev.error}")
            else:
                print(f"  [todo] {key}")

    if args.json_out is not None:
        payload = {
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "total_events": summary.total_events,
            "completed_events": summary.completed_events,
            "partial_events": summary.partial_events,
            "not_started_events": summary.not_started_events,
            "error_events": summary.error_events,
            "completed_duration_sec": summary.completed_duration_sec,
            "completed_span_count": summary.completed_span_count,
            "by_sport_total": summary.by_sport_total,
            "by_sport_completed": summary.by_sport_completed,
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

