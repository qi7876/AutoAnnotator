#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


_NUMERIC = re.compile(r"^\d+$")


@dataclass(frozen=True)
class Stats:
    scanned_events: int = 0
    deleted_root_longvideo: int = 0
    moved_pairs: int = 0
    skipped_existing_dest: int = 0
    missing_pair: int = 0
    failed: int = 0


def _iter_events(dataset_root: Path):
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue
        for event_dir in sorted(sport_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            yield sport_dir, event_dir


def _is_numeric_stem(path: Path) -> bool:
    return bool(_NUMERIC.match(path.stem))


def _collect_event_root_files(
    event_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    mp4s: dict[str, Path] = {}
    jsons: dict[str, Path] = {}

    for p in event_dir.iterdir():
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix == ".mp4" and _is_numeric_stem(p):
            mp4s[p.stem] = p
        elif suffix == ".json" and _is_numeric_stem(p):
            jsons[p.stem] = p

    return mp4s, jsons


def _safe_unlink(path: Path, apply: bool) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        return False

    if apply:
        path.unlink()
    return True


def _move_pair_two_phase(
    *,
    src_mp4: Path,
    src_json: Path,
    dst_mp4: Path,
    dst_json: Path,
    apply: bool,
) -> None:
    if not apply:
        return

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex[:10]
    tmp_mp4 = dst_mp4.parent / f".tmp_{token}_{dst_mp4.name}"
    tmp_json = dst_json.parent / f".tmp_{token}_{dst_json.name}"

    shutil.move(str(src_mp4), str(tmp_mp4))
    shutil.move(str(src_json), str(tmp_json))

    tmp_mp4.rename(dst_mp4)
    tmp_json.rename(dst_json)


def convert_dataset(
    *,
    dataset_root: Path,
    apply: bool,
    delete_longvideo: bool,
    overwrite: bool,
    start_index: int,
    sport: str | None,
    event: str | None,
    max_events: int | None,
) -> Stats:
    stats = Stats()

    def _with(**kwargs) -> Stats:
        return Stats(**{**stats.__dict__, **kwargs})

    events_done = 0

    for sport_dir, event_dir in _iter_events(dataset_root):
        if sport is not None and sport_dir.name != sport:
            continue
        if event is not None and event_dir.name != event:
            continue

        stats = _with(scanned_events=stats.scanned_events + 1)

        if delete_longvideo:
            for name in ("1.mp4", "1.json"):
                if _safe_unlink(event_dir / name, apply=apply):
                    stats = _with(
                        deleted_root_longvideo=stats.deleted_root_longvideo + 1
                    )

        mp4s, jsons = _collect_event_root_files(event_dir)

        mp4s.pop("1", None)
        jsons.pop("1", None)

        stems = sorted(set(mp4s.keys()) | set(jsons.keys()), key=lambda s: int(s))
        pairs: list[tuple[str, Path, Path]] = []

        for stem in stems:
            m = mp4s.get(stem)
            j = jsons.get(stem)
            if m is None or j is None:
                stats = _with(missing_pair=stats.missing_pair + 1)
                continue
            pairs.append((stem, m, j))

        if not pairs:
            events_done += 1
            if max_events is not None and events_done >= max_events:
                break
            continue

        clips_dir = event_dir / "clips"
        if apply:
            clips_dir.mkdir(parents=True, exist_ok=True)

        idx = start_index
        for _, src_mp4, src_json in pairs:
            dst_mp4 = clips_dir / f"{idx}.mp4"
            dst_json = clips_dir / f"{idx}.json"

            if dst_mp4.exists() or dst_json.exists():
                if not overwrite:
                    stats = _with(skipped_existing_dest=stats.skipped_existing_dest + 1)
                    idx += 1
                    continue
                if apply:
                    if dst_mp4.exists():
                        dst_mp4.unlink()
                    if dst_json.exists():
                        dst_json.unlink()

            try:
                _move_pair_two_phase(
                    src_mp4=src_mp4,
                    src_json=src_json,
                    dst_mp4=dst_mp4,
                    dst_json=dst_json,
                    apply=apply,
                )
                stats = _with(moved_pairs=stats.moved_pairs + 1)
            except Exception:
                stats = _with(failed=stats.failed + 1)

            idx += 1

        events_done += 1
        if max_events is not None and events_done >= max_events:
            break

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert event-root numeric clip assets into clips/ with renumbering. "
            "Deletes event-root 1.mp4/1.json by default. Dry-run unless --apply."
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
        help="Perform changes (default: dry-run)",
    )
    parser.add_argument(
        "--keep-longvideo",
        action="store_true",
        help="Do not delete event-root 1.mp4/1.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files in clips/ if they exist",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start index for renumbering (default: 1)",
    )
    parser.add_argument(
        "--sport",
        default=None,
        help="Only process a specific sport directory name",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Only process a specific event directory name",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after N events (debug)",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    if args.start_index < 1:
        print("--start-index must be >= 1", file=sys.stderr)
        return 2

    stats = convert_dataset(
        dataset_root=dataset_root,
        apply=bool(args.apply),
        delete_longvideo=not bool(args.keep_longvideo),
        overwrite=bool(args.overwrite),
        start_index=int(args.start_index),
        sport=args.sport,
        event=args.event,
        max_events=args.max_events,
    )

    print("dataset_root:", dataset_root)
    print("apply       :", bool(args.apply))
    print("delete_1.*  :", not bool(args.keep_longvideo))
    print("overwrite   :", bool(args.overwrite))
    print("start_index :", args.start_index)
    if args.sport is not None:
        print("sport filter:", args.sport)
    if args.event is not None:
        print("event filter:", args.event)

    print("scanned_events       :", stats.scanned_events)
    if args.apply:
        print("deleted_root_1_files :", stats.deleted_root_longvideo)
        print("moved_pairs          :", stats.moved_pairs)
    else:
        print("would_delete_1_files :", stats.deleted_root_longvideo)
        print("would_move_pairs     :", stats.moved_pairs)
    print("missing_pair         :", stats.missing_pair)
    print("skipped_existing_dest:", stats.skipped_existing_dest)
    print("failed               :", stats.failed)

    if stats.failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
