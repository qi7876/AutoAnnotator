#!/usr/bin/env python3
"""Summarize video file sizes under the repository data directory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median


VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}
)


@dataclass(frozen=True)
class VideoFileSize:
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class ExtensionSummary:
    extension: str
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class VideoSizeSummary:
    root: Path
    file_count: int
    total_bytes: int
    average_bytes: float
    median_bytes: float
    min_file: VideoFileSize | None
    max_file: VideoFileSize | None
    by_extension: list[ExtensionSummary]
    oversized_files: list[VideoFileSize]


def format_bytes(size_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def iter_video_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def collect_video_sizes(root: Path) -> list[VideoFileSize]:
    return [VideoFileSize(path=path, size_bytes=path.stat().st_size) for path in iter_video_files(root)]


def build_summary(
    *,
    root: Path,
    files: list[VideoFileSize],
    max_size_bytes: int | None = None,
) -> VideoSizeSummary:
    sorted_files = sorted(files, key=lambda item: item.size_bytes)
    total_bytes = sum(item.size_bytes for item in sorted_files)
    sizes = [item.size_bytes for item in sorted_files]

    extension_totals: dict[str, tuple[int, int]] = {}
    for item in sorted_files:
        extension = item.path.suffix.lower() or "<none>"
        count, total = extension_totals.get(extension, (0, 0))
        extension_totals[extension] = (count + 1, total + item.size_bytes)

    by_extension = [
        ExtensionSummary(extension=extension, file_count=count, total_bytes=total)
        for extension, (count, total) in sorted(
            extension_totals.items(),
            key=lambda item: (-item[1][1], item[0]),
        )
    ]

    oversized_files = []
    if max_size_bytes is not None:
        oversized_files = [
            item for item in sorted(sorted_files, key=lambda entry: (-entry.size_bytes, str(entry.path)))
            if item.size_bytes > max_size_bytes
        ]

    return VideoSizeSummary(
        root=root,
        file_count=len(sorted_files),
        total_bytes=total_bytes,
        average_bytes=(total_bytes / len(sorted_files)) if sorted_files else 0.0,
        median_bytes=float(median(sizes)) if sizes else 0.0,
        min_file=sorted_files[0] if sorted_files else None,
        max_file=sorted_files[-1] if sorted_files else None,
        by_extension=by_extension,
        oversized_files=oversized_files,
    )


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(prog="check-video-sizes")
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "data",
        help="Directory to scan recursively (default: repo_root/data).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many largest files to print (default: 10). Use 0 to disable.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print every matched video file sorted by size descending.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help="Flag files larger than this size in MiB and return exit code 1 if any are found.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the summary as JSON.",
    )
    return parser


def _print_file_list(title: str, files: list[VideoFileSize], *, root: Path) -> None:
    if not files:
        return
    print("")
    print(title)
    for item in files:
        print(f"  {format_bytes(item.size_bytes):>10}  {_display_path(item.path, root)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"Data root not found or not a directory: {root}")
        return 2

    max_size_bytes = None
    if args.max_size_mb is not None:
        if args.max_size_mb < 0:
            print("--max-size-mb must be >= 0.")
            return 2
        max_size_bytes = int(args.max_size_mb * 1024 * 1024)

    files = collect_video_sizes(root)
    summary = build_summary(root=root, files=files, max_size_bytes=max_size_bytes)

    print("Video file size summary")
    print(f"  Root: {root}")
    print(f"  Files: {summary.file_count}")
    print(f"  Total size: {format_bytes(summary.total_bytes)}")
    print(f"  Average size: {format_bytes(summary.average_bytes)}")
    print(f"  Median size: {format_bytes(summary.median_bytes)}")

    if summary.max_file is not None:
        print(
            "  Largest: "
            f"{format_bytes(summary.max_file.size_bytes)}  "
            f"{_display_path(summary.max_file.path, root)}"
        )
    if summary.min_file is not None:
        print(
            "  Smallest: "
            f"{format_bytes(summary.min_file.size_bytes)}  "
            f"{_display_path(summary.min_file.path, root)}"
        )

    if summary.by_extension:
        print("")
        print("By extension")
        for item in summary.by_extension:
            print(
                f"  {item.extension}: {item.file_count} files, {format_bytes(item.total_bytes)}"
            )

    if args.top > 0 and files:
        top_files = sorted(files, key=lambda item: (-item.size_bytes, str(item.path)))[: args.top]
        _print_file_list(f"Top {len(top_files)} largest files", top_files, root=root)

    if args.details and files:
        all_files = sorted(files, key=lambda item: (-item.size_bytes, str(item.path)))
        _print_file_list("All matched video files", all_files, root=root)

    if args.max_size_mb is not None:
        threshold_label = format_bytes(max_size_bytes or 0)
        print("")
        print(
            f"Files over {threshold_label}: {len(summary.oversized_files)}"
        )
        for item in summary.oversized_files:
            print(f"  {format_bytes(item.size_bytes):>10}  {_display_path(item.path, root)}")

    if args.json_out is not None:
        payload = {
            "root": str(root),
            "file_count": summary.file_count,
            "total_bytes": summary.total_bytes,
            "average_bytes": summary.average_bytes,
            "median_bytes": summary.median_bytes,
            "min_file": (
                {
                    "path": _display_path(summary.min_file.path, root),
                    "size_bytes": summary.min_file.size_bytes,
                }
                if summary.min_file is not None
                else None
            ),
            "max_file": (
                {
                    "path": _display_path(summary.max_file.path, root),
                    "size_bytes": summary.max_file.size_bytes,
                }
                if summary.max_file is not None
                else None
            ),
            "by_extension": [
                {
                    "extension": item.extension,
                    "file_count": item.file_count,
                    "total_bytes": item.total_bytes,
                }
                for item in summary.by_extension
            ],
            "oversized_files": [
                {
                    "path": _display_path(item.path, root),
                    "size_bytes": item.size_bytes,
                }
                for item in summary.oversized_files
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 1 if summary.oversized_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
