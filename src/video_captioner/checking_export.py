"""Export completed Video Captioner outputs for manual checking."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .pipeline import resolve_dataset_root
from .progress import EventKey, collect_progress

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


FILES_TO_COPY = ("segment.mp4", "run_meta.json", "long_caption.json")


@dataclass(frozen=True)
class ExportReport:
    dest_root: Path
    completed_events: list[EventKey]
    copied_files: int
    skipped_files: int
    missing_files: int


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def export_caption_checking(
    *,
    dataset_root: Path,
    output_root: Path,
    dest_root: Path,
    overwrite: bool = False,
    progress: bool | None = None,
) -> ExportReport:
    """
    Copy each completed sport/event's key outputs into `dest_root`, preserving structure.

    We define "completed" as events with a valid `long_caption.json` (schema-validated).
    """
    dataset_root = resolve_dataset_root(dataset_root)
    summary, events = collect_progress(dataset_root=dataset_root, output_root=output_root)
    completed = [ev for ev in events if ev.status == "completed"]
    completed_keys = [EventKey(sport=ev.key.sport, event=ev.key.event) for ev in completed]

    dest_root.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    skipped_files = 0
    missing_files = 0

    iterator = completed_keys
    progress_on = sys.stderr.isatty() if progress is None else bool(progress)
    if tqdm is not None and progress_on:
        iterator = tqdm(iterator, desc="Export", unit="event")  # type: ignore[assignment]

    for key in iterator:
        src_dir = output_root / key.sport / key.event
        dst_dir = dest_root / key.sport / key.event
        dst_dir.mkdir(parents=True, exist_ok=True)

        for name in FILES_TO_COPY:
            src = src_dir / name
            if not src.is_file():
                missing_files += 1
                logger.warning(f"Missing expected file (skipping): {src}")
                continue

            dst = dst_dir / name
            if dst.exists() and not overwrite:
                skipped_files += 1
                continue
            shutil.copy2(src, dst)
            copied_files += 1

    _write_json(
        dest_root / "completed_events.json",
        {
            "total_events": summary.total_events,
            "completed_events": summary.completed_events,
            "completed_duration_sec": summary.completed_duration_sec,
            "completed_span_count": summary.completed_span_count,
            "events": [{"sport": k.sport, "event": k.event} for k in completed_keys],
        },
    )
    (dest_root / "completed_events.txt").write_text(
        "\n".join([f"{k.sport}/{k.event}" for k in completed_keys]) + "\n",
        encoding="utf-8",
    )

    return ExportReport(
        dest_root=dest_root,
        completed_events=completed_keys,
        copied_files=copied_files,
        skipped_files=skipped_files,
        missing_files=missing_files,
    )
