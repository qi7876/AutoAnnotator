#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CLIP_EXTS: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi")
FRAME_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class SegmentEntry:
    json_path: Path
    kind: str
    segment_ids: tuple[str, ...]
    state: str  # "empty" | "non_empty" | "unknown"


@dataclass(frozen=True)
class Change:
    json_path: Path
    deleted_media: tuple[str, ...]


@dataclass(frozen=True)
class Stats:
    scanned_events: int = 0
    eligible_events: int = 0
    skipped_all_empty_events: int = 0
    scanned_jsons: int = 0
    deleted_jsons: int = 0
    deleted_media_files: int = 0
    invalid_jsons: int = 0
    failed_deletes: int = 0


def _iter_events(dataset_root: Path) -> Iterable[Path]:
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue
        for event_dir in sorted(sport_dir.iterdir()):
            if event_dir.is_dir():
                yield event_dir


def _iter_segment_jsons(event_dir: Path) -> Iterable[Path]:
    yield from sorted((event_dir / "clips").glob("*.json"))
    yield from sorted((event_dir / "frames").glob("*.json"))


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _parse_task_items(data: dict[str, Any]) -> tuple[str, ...] | None:
    for key in ("task_to_annotate", "tasks_to_annotate"):
        raw = data.get(key)
        if not isinstance(raw, list):
            continue
        tasks: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if name:
                tasks.append(name)
        return tuple(tasks)
    return None


def _safe_segment_id(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None

    if not text:
        return None
    if "/" in text or "\\" in text:
        return None
    return text


def _build_entry(path: Path) -> tuple[SegmentEntry | None, int]:
    data = _read_json_dict(path)
    if data is None:
        return None, 1

    kind = path.parent.name
    ids: list[str] = [path.stem]
    from_id = _safe_segment_id(data.get("id"))
    if from_id and from_id not in ids:
        ids.append(from_id)

    parsed_tasks = _parse_task_items(data)
    if parsed_tasks is None:
        state = "unknown"
    elif len(parsed_tasks) == 0:
        state = "empty"
    else:
        state = "non_empty"

    return SegmentEntry(json_path=path, kind=kind, segment_ids=tuple(ids), state=state), 0


def _candidate_media_paths(entry: SegmentEntry) -> list[Path]:
    base = entry.json_path.parent
    candidates: list[Path] = []
    if entry.kind == "clips":
        exts = CLIP_EXTS
    elif entry.kind == "frames":
        exts = FRAME_EXTS
    else:
        return candidates

    for segment_id in entry.segment_ids:
        for ext in exts:
            candidates.append(base / f"{segment_id}{ext}")
    return candidates


def cleanup_empty_task_segments(
    *,
    dataset_root: Path,
    apply: bool,
    sport: str | None = None,
    event: str | None = None,
) -> tuple[list[Change], Stats]:
    changes: list[Change] = []
    stats = Stats()

    def _with(**kwargs: int) -> None:
        nonlocal stats
        stats = Stats(**{**stats.__dict__, **kwargs})

    for event_dir in _iter_events(dataset_root):
        sport_name = event_dir.parent.name
        event_name = event_dir.name
        if sport is not None and sport_name != sport:
            continue
        if event is not None and event_name != event:
            continue

        _with(scanned_events=stats.scanned_events + 1)

        entries: list[SegmentEntry] = []
        invalid_count = 0
        for json_path in _iter_segment_jsons(event_dir):
            entry, invalid = _build_entry(json_path)
            invalid_count += invalid
            if entry is not None:
                entries.append(entry)

        _with(
            scanned_jsons=stats.scanned_jsons + len(entries),
            invalid_jsons=stats.invalid_jsons + invalid_count,
        )

        if not entries:
            continue

        has_non_empty = any(entry.state in ("non_empty", "unknown") for entry in entries)
        if not has_non_empty:
            _with(skipped_all_empty_events=stats.skipped_all_empty_events + 1)
            continue

        _with(eligible_events=stats.eligible_events + 1)

        for entry in entries:
            if entry.state != "empty":
                continue

            deleted_media: list[str] = []
            if apply:
                try:
                    entry.json_path.unlink()
                    _with(deleted_jsons=stats.deleted_jsons + 1)
                except OSError:
                    _with(failed_deletes=stats.failed_deletes + 1)
                    continue
            else:
                _with(deleted_jsons=stats.deleted_jsons + 1)

            for media_path in _candidate_media_paths(entry):
                if not media_path.exists():
                    continue
                if apply:
                    try:
                        media_path.unlink()
                        _with(deleted_media_files=stats.deleted_media_files + 1)
                        deleted_media.append(str(media_path))
                    except OSError:
                        _with(failed_deletes=stats.failed_deletes + 1)
                else:
                    _with(deleted_media_files=stats.deleted_media_files + 1)
                    deleted_media.append(str(media_path))

            changes.append(Change(json_path=entry.json_path, deleted_media=tuple(deleted_media)))

    return changes, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete segment JSONs whose task_to_annotate/tasks_to_annotate is empty, "
            "plus corresponding segment media. If an event has only empty-task JSONs, skip that event."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root (default: data/Dataset)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions in-place (default: dry-run)",
    )
    parser.add_argument(
        "--sport",
        default=None,
        help="Only process one sport directory",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Only process one event directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print changed entries as JSON lines",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"dataset_root not found or not a directory: {dataset_root}", file=sys.stderr)
        return 2

    changes, stats = cleanup_empty_task_segments(
        dataset_root=dataset_root,
        apply=bool(args.apply),
        sport=args.sport,
        event=args.event,
    )

    if args.json:
        for change in changes:
            print(
                json.dumps(
                    {
                        "json_path": str(change.json_path),
                        "deleted_media": list(change.deleted_media),
                    },
                    ensure_ascii=False,
                )
            )
    else:
        print("dataset_root:", dataset_root)
        print("apply:", bool(args.apply))
        if args.sport is not None:
            print("sport filter:", args.sport)
        if args.event is not None:
            print("event filter:", args.event)
        print("scanned_events:", stats.scanned_events)
        print("eligible_events:", stats.eligible_events)
        print("skipped_all_empty_events:", stats.skipped_all_empty_events)
        print("scanned_jsons:", stats.scanned_jsons)
        print("invalid_jsons:", stats.invalid_jsons)
        if args.apply:
            print("deleted_jsons:", stats.deleted_jsons)
            print("deleted_media_files:", stats.deleted_media_files)
            print("failed_deletes:", stats.failed_deletes)
        else:
            print("would_delete_jsons:", stats.deleted_jsons)
            print("would_delete_media_files:", stats.deleted_media_files)
            if changes:
                print("\nRun with --apply to write changes.")

    if stats.failed_deletes > 0:
        return 2
    return 1 if changes and not args.apply else 0


if __name__ == "__main__":
    raise SystemExit(main())
