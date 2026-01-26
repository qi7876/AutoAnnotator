#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CopyStats:
    scanned_events: int = 0
    found: int = 0
    transferred: int = 0
    skipped_exists: int = 0
    missing: int = 0
    failed: int = 0
    json_found: int = 0
    json_transferred: int = 0
    json_skipped_exists: int = 0
    json_missing: int = 0
    json_failed: int = 0


def _iter_event_dirs(dataset_root: Path):
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue
        for event_dir in sorted(sport_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            yield sport_dir, event_dir


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _ensure_parent_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path, *, mode: str, dry_run: bool) -> None:
    if dry_run:
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode in ("move", "mv"):
        shutil.move(str(src), str(dst))
        return

    if mode == "hardlink":
        os.link(src, dst)
        return

    if mode == "symlink":
        rel_src = os.path.relpath(src, start=dst.parent)
        os.symlink(rel_src, dst)
        return

    raise ValueError(f"Unknown mode: {mode}")


def build_subset(
    *,
    dataset_root: Path,
    output_root: Path,
    filename: str,
    with_json: bool,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    sport: str | None,
    event: str | None,
    max_actions: int | None,
    print_missing: bool,
) -> CopyStats:
    stats = CopyStats()

    def _with(**kwargs) -> CopyStats:
        return CopyStats(**{**stats.__dict__, **kwargs})

    actions_done = 0
    src_rel_path = Path(filename)
    json_rel_path = src_rel_path.with_suffix(".json")

    def _transfer(src_path: Path, dst_path: Path) -> bool:
        try:
            _ensure_parent_dir(dst_path, dry_run)
            _copy_file(src_path, dst_path, mode=mode, dry_run=dry_run)
            return True
        except OSError as e:
            if mode == "hardlink" and getattr(e, "errno", None) == getattr(
                os, "EXDEV", 18
            ):
                try:
                    _copy_file(src_path, dst_path, mode="copy", dry_run=dry_run)
                    return True
                except Exception:
                    return False
            return False

    for sport_dir, event_dir in _iter_event_dirs(dataset_root):
        if sport is not None and sport_dir.name != sport:
            continue
        if event is not None and event_dir.name != event:
            continue

        stats = _with(scanned_events=stats.scanned_events + 1)

        src_path = event_dir / filename
        if not src_path.is_file():
            stats = _with(missing=stats.missing + 1)
            if print_missing:
                print(src_path)
            continue

        dst_path = output_root / sport_dir.name / event_dir.name / filename

        stats = _with(found=stats.found + 1)

        if dst_path.exists():
            if not overwrite:
                stats = _with(skipped_exists=stats.skipped_exists + 1)
                continue
            _safe_unlink(dst_path)

        if _transfer(src_path, dst_path):
            stats = _with(transferred=stats.transferred + 1)
        else:
            stats = _with(failed=stats.failed + 1)

        if with_json and json_rel_path != src_rel_path:
            json_src = event_dir / json_rel_path
            if not json_src.is_file():
                stats = _with(json_missing=stats.json_missing + 1)
                if print_missing:
                    print(json_src)
            else:
                stats = _with(json_found=stats.json_found + 1)
                json_dst = (
                    output_root / sport_dir.name / event_dir.name / json_rel_path
                )

                if json_dst.exists():
                    if not overwrite:
                        stats = _with(json_skipped_exists=stats.json_skipped_exists + 1)
                    else:
                        _safe_unlink(json_dst)
                        if _transfer(json_src, json_dst):
                            stats = _with(
                                json_transferred=stats.json_transferred + 1
                            )
                        else:
                            stats = _with(json_failed=stats.json_failed + 1)
                else:
                    if _transfer(json_src, json_dst):
                        stats = _with(json_transferred=stats.json_transferred + 1)
                    else:
                        stats = _with(json_failed=stats.json_failed + 1)

        actions_done += 1
        if max_actions is not None and actions_done >= max_actions:
            break

    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Source dataset root (default: data/Dataset)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/waibao"),
        help="Output subset root (default: data/waibao)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output directory name under data/ (e.g. waibao2). Use either --output-root or --output-name.",
    )
    parser.add_argument(
        "--filename",
        default="1.mp4",
        help="Which file to extract from each event directory (default: 1.mp4)",
    )
    parser.add_argument(
        "--with-json",
        action="store_true",
        help='Also export the sibling metadata JSON (e.g., "1.mp4" -> "1.json")',
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move", "mv", "hardlink", "symlink"),
        default="copy",
        help="How to place files into the subset (default: copy)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write anything; only scan and report",
    )
    parser.add_argument(
        "--print-missing",
        action="store_true",
        help="Print each missing source path (event_dir/filename)",
    )
    parser.add_argument(
        "--sport",
        default=None,
        help="Only process a specific sport directory name (optional)",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="Only process a specific event directory name (optional)",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Stop after N successful actions (useful for quick sanity checks)",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root

    default_output_root = Path("data/waibao")
    output_root = args.output_root
    if args.output_name is not None:
        if output_root != default_output_root:
            raise SystemExit("Use either --output-root or --output-name")
        name = str(args.output_name).strip()
        if not name:
            raise SystemExit("--output-name must be non-empty")
        name_path = Path(name)
        if name_path.is_absolute() or name_path.name != name:
            raise SystemExit("--output-name must be a directory name, not a path")
        output_root = Path("data") / name

    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root is not a directory: {dataset_root}")

    stats = build_subset(
        dataset_root=dataset_root,
        output_root=output_root,
        filename=args.filename,
        with_json=bool(args.with_json),
        mode=args.mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        sport=args.sport,
        event=args.event,
        max_actions=args.max_actions,
        print_missing=bool(args.print_missing),
    )

    print("dataset_root:", dataset_root)
    print("output_root :", output_root)
    print("filename    :", args.filename)
    print("with_json   :", bool(args.with_json))
    print("mode        :", args.mode)
    print("dry_run     :", args.dry_run)
    print("overwrite   :", args.overwrite)
    if args.sport is not None:
        print("sport filter:", args.sport)
    if args.event is not None:
        print("event filter:", args.event)

    print()
    print("scanned_events:", stats.scanned_events)
    print("found        :", stats.found)
    print("transferred  :", stats.transferred)
    print("skipped_exists:", stats.skipped_exists)
    print("missing      :", stats.missing)
    print("failed       :", stats.failed)
    if bool(args.with_json):
        print("json_found   :", stats.json_found)
        print("json_transferred:", stats.json_transferred)
        print("json_skipped_exists:", stats.json_skipped_exists)
        print("json_missing :", stats.json_missing)
        print("json_failed  :", stats.json_failed)

    if stats.failed > 0 or (bool(args.with_json) and stats.json_failed > 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
