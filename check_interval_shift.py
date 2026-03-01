#!/usr/bin/env python3
"""Verify that MOT txt frame indices match annotation intervals shifted by one."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class CheckResult:
    json_path: Path
    annotation_id: str
    mot_path: Path | None
    status: str
    details: str
    category: str | None = None


def as_rel_path(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch check whether MOT txt frame ranges match the annotation intervals "
            "from each JSON file, shifted by +1 frame."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="output",
        help="Dataset root to scan (default: %(default)s within the current working directory).",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help=(
            "Workspace base used to resolve relative mot_file paths (default: current "
            "working directory)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print mismatches and errors.",
    )
    parser.add_argument(
        "--report",
        default="interval_check_report.json",
        help="Path to the JSON report collecting start/end discrepancies.",
    )
    return parser.parse_args()


def gather_json_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def parse_intervals(raw: Sequence) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    if not isinstance(raw, Sequence):
        return intervals

    # Handle plain numeric sequences like [0, 36]
    if raw and all(isinstance(v, (int, float)) for v in raw):
        if len(raw) % 2 != 0:
            raise ValueError(f"Odd number of numeric entries in interval {raw}")
        it = iter(raw)
        for start in it:
            end = next(it)
            intervals.append((int(start), int(end)))
        return intervals

    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
            intervals.append((int(start), int(end)))
        elif isinstance(item, str) and "-" in item:
            left, right = item.split("-", 1)
            intervals.append((int(left.strip()), int(right.strip())))
        elif isinstance(item, (int, float)):
            raise ValueError(
                "Standalone numeric value found inside interval list; expected pairs."
            )
    return intervals


def expand_frames(intervals: Iterable[Tuple[int, int]], shift: int = 1) -> List[int]:
    frames = set()
    for start, end in intervals:
        if end < start:
            raise ValueError(f"Interval start {start} exceeds end {end}.")
        frames.update(range(int(start + shift), int(end + shift) + 1))
    return sorted(frames)


def read_mot_frames(path: Path) -> List[int]:
    frames = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if not parts:
                continue
            try:
                frame_idx = int(float(parts[0]))
            except ValueError:
                continue
            frames.add(frame_idx)
    return sorted(frames)


def resolve_mot_path(json_path: Path, mot_entry: str, workspace: Path) -> Path | None:
    candidates = []
    entry_path = Path(mot_entry)
    if entry_path.is_absolute():
        candidates.append(entry_path)
    else:
        candidates.append(json_path.parent / entry_path)
        candidates.append(workspace / entry_path)
        if entry_path.parts and entry_path.parts[0].lower() == "data":
            trimmed = Path(*entry_path.parts[1:])
            candidates.append(json_path.parent / trimmed)
            candidates.append(workspace / trimmed)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def compare_frames(expected: List[int], actual: List[int]) -> Tuple[bool, str, str | None]:
    expected_set = set(expected)
    actual_set = set(actual)
    if expected_set == actual_set:
        if not expected_set:
            return True, "No frames defined in annotation.", None
        return True, (
            f"frames {min(expected_set)}-{max(expected_set)} align (len={len(expected_set)})"
        ), None

    if expected_set and actual_set:
        exp_min = min(expected_set)
        exp_max = max(expected_set)
        act_min = min(actual_set)
        act_max = max(actual_set)
        if act_min != exp_min:
            return False, f"start mismatch expected {exp_min}, got {act_min}", "start_mismatch"
        if act_max > exp_max:
            return False, f"end exceeds expected max {exp_max}, got {act_max}", "end_long"
        if act_min == exp_min and act_max == exp_max and len(actual_set) != len(expected_set):
            missing = sorted(expected_set - actual_set)
            detail = (
                "frame count mismatch: annotation and MOT cover same bounds but differ in frame"
                " count (likely gaps inside the window)."
            )
            if missing:
                detail = f"{detail} Missing {summarize_span(missing)}."
            return False, detail, "count_mismatch"
        if actual_set.issubset(expected_set):
            if act_max < exp_max:
                return (
                    True,
                    f"frames {act_min}-{act_max} align (len={len(actual_set)}) (allowed truncated/missing middle)",
                    "end_short",
                )
            return True, (
                f"frames {act_min}-{act_max} align (len={len(actual_set)})"
            ), None

    if expected_set and not actual_set:
        exp_min = min(expected_set)
        exp_max = max(expected_set)
        return False, f"tracking empty (expected {exp_min}-{exp_max})", "end_short"

    missing = sorted(expected_set - actual_set)
    extras = sorted(actual_set - expected_set)
    detail_parts = []
    if missing:
        detail_parts.append(f"missing {summarize_span(missing)}")
    if extras:
        detail_parts.append(f"extra {summarize_span(extras)}")
    return False, ", ".join(detail_parts), None


def summarize_span(values: Sequence[int], limit: int = 6) -> str:
    if not values:
        return "none"
    if len(values) <= limit:
        return ",".join(str(v) for v in values)
    return f"{values[0]}-{values[limit-1]} (+{len(values) - limit} more)"


def process_json(json_path: Path, workspace: Path) -> List[CheckResult]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return [
            CheckResult(
                json_path=json_path,
                annotation_id="*",
                mot_path=None,
                status="error",
                details=f"Failed to parse JSON: {exc}",
            )
        ]

    annotations = payload.get("annotations", [])
    results: List[CheckResult] = []
    for annotation in annotations:
        ann_id = str(annotation.get("annotation_id", "?"))
        frames_raw = annotation.get("A_window_frame")
        tracking = annotation.get("tracking_bboxes")
        if not tracking:
            results.append(
                CheckResult(
                    json_path=json_path,
                    annotation_id=ann_id,
                    mot_path=None,
                    status="skipped",
                    details="No tracking_bboxes entry.",
                )
            )
            continue
        mot_entry = tracking.get("mot_file")
        if not mot_entry:
            results.append(
                CheckResult(
                    json_path=json_path,
                    annotation_id=ann_id,
                    mot_path=None,
                    status="skipped",
                    details="tracking_bboxes missing mot_file.",
                )
            )
            continue
        mot_path = resolve_mot_path(json_path, mot_entry, workspace)
        if mot_path is None:
            results.append(
                CheckResult(
                    json_path=json_path,
                    annotation_id=ann_id,
                    mot_path=None,
                    status="error",
                    details=f"mot_file not found: {mot_entry}",
                )
            )
            continue

        try:
            intervals = parse_intervals(frames_raw or [])
            expected_frames = expand_frames(intervals)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            results.append(
                CheckResult(
                    json_path=json_path,
                    annotation_id=ann_id,
                    mot_path=mot_path,
                    status="error",
                    details=f"Failed to parse annotation intervals: {exc}",
                )
            )
            continue

        try:
            actual_frames = read_mot_frames(mot_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            results.append(
                CheckResult(
                    json_path=json_path,
                    annotation_id=ann_id,
                    mot_path=mot_path,
                    status="error",
                    details=f"Failed to read MOT file: {exc}",
                )
            )
            continue

        ok, detail, category = compare_frames(expected_frames, actual_frames)
        results.append(
            CheckResult(
                json_path=json_path,
                annotation_id=ann_id,
                mot_path=mot_path,
                status="ok" if ok else "mismatch",
                details=detail,
                category=category,
            )
        )
    return results


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.root).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    if not dataset_root.exists():
        print(f"Dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2
    rel_base = dataset_root.parent

    json_files = gather_json_files(dataset_root)
    if not json_files:
        print(f"No JSON files found under {dataset_root}", file=sys.stderr)
        return 2

    total = 0
    mismatches = 0
    errors = 0
    skipped = 0
    category_records = {
        "start_mismatch": [],
        "end_short": [],
        "end_long": [],
        "count_mismatch": [],
    }
    for json_path in json_files:
        for result in process_json(json_path, workspace):
            total += 1
            rel_json = as_rel_path(result.json_path, rel_base)
            rel_mot = as_rel_path(result.mot_path, rel_base) if result.mot_path else None

            if result.category:
                category_records.setdefault(result.category, []).append(
                    {
                        "json_path": rel_json,
                        "annotation_id": result.annotation_id,
                        "mot_path": rel_mot,
                        "detail": result.details,
                    }
                )

            if result.status == "ok":
                continue
            if result.status == "mismatch":
                mismatches += 1
                print(
                    f"MISMATCH {rel_json}:{result.annotation_id} ({rel_mot}) -> {result.details}"
                )
            elif result.status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(
                    f"ERROR {json_path}:{result.annotation_id}: {result.details}",
                    file=sys.stderr,
                )

    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(category_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"Report written to {report_path}")

    count_mismatch_entries = category_records.get("count_mismatch", [])
    if count_mismatch_entries:
        print("Count-mismatch annotations:")
        for entry in count_mismatch_entries:
            print(
                f"  {entry['json_path']}:{entry['annotation_id']} ({entry['mot_path']}) -> {entry['detail']}"
            )

    print(
        f"Checked {total} annotations | mismatches={mismatches} errors={errors} skipped={skipped}",
        file=sys.stderr,
    )
    return 1 if mismatches or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
