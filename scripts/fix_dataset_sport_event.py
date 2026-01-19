#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FixResult:
    path: Path
    expected_sport: str
    expected_event: str
    fixed_origin: bool
    fixed_top_level: bool


def _iter_dataset_jsons(dataset_root: Path) -> Iterable[Path]:
    yield from sorted(dataset_root.glob("*/*/frames/*.json"))
    yield from sorted(dataset_root.glob("*/*/clips/*.json"))
    yield from sorted(dataset_root.glob("*/*/*.json"))


def _infer_sport_event(dataset_root: Path, path: Path) -> tuple[str, str] | None:
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        return None

    if len(rel.parts) < 3:
        return None

    sport, event = rel.parts[0], rel.parts[1]
    if not sport or not event:
        return None

    return sport, event


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def _set_str_key(d: dict[str, Any], key: str, value: str) -> bool:
    current = d.get(key)
    if current != value:
        d[key] = value
        return True
    return False


def fix_one(data: Any, sport: str, event: str) -> tuple[Any, bool, bool]:
    if not isinstance(data, dict):
        return data, False, False

    fixed_origin = False
    fixed_top = False

    origin = data.get("origin")
    if isinstance(origin, dict):
        fixed_origin |= _set_str_key(origin, "sport", sport)
        fixed_origin |= _set_str_key(origin, "event", event)
        data["origin"] = origin

    if "sport" in data:
        v = data.get("sport")
        if isinstance(v, str) or v is None:
            fixed_top |= _set_str_key(data, "sport", sport)

    if "event" in data:
        v = data.get("event")
        if isinstance(v, str) or v is None:
            fixed_top |= _set_str_key(data, "event", event)

    if ("origin" not in data) and ("id" in data) and ("tasks_to_annotate" in data):
        data["origin"] = {"sport": sport, "event": event}
        fixed_origin = True

    return data, fixed_origin, fixed_top


def _get_declared_sport_event(data: Any) -> tuple[str | None, str | None]:
    if not isinstance(data, dict):
        return None, None

    origin = data.get("origin")
    if isinstance(origin, dict):
        s = origin.get("sport")
        e = origin.get("event")
        if isinstance(s, str) and isinstance(e, str) and s and e:
            return s, e

    s = data.get("sport")
    e = data.get("event")
    if isinstance(s, str) and isinstance(e, str) and s and e:
        return s, e

    return None, None


def _get_segment_id(data: Any, default: str) -> str:
    if isinstance(data, dict):
        v = data.get("id")
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
    return default


def _extract_mot_refs(output_data: Any) -> set[str]:
    refs: set[str] = set()
    if not isinstance(output_data, dict):
        return refs

    annotations = output_data.get("annotations")
    if not isinstance(annotations, list):
        return refs

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        tracking = ann.get("tracking_bboxes")
        if not isinstance(tracking, dict):
            continue
        mot = tracking.get("mot_file")
        if isinstance(mot, str) and mot.strip():
            refs.add(mot.strip())

    return refs


def _resolve_mot_path(project_root: Path, mot_ref: str) -> Path:
    p = Path(mot_ref)
    if p.is_absolute():
        return p
    return project_root / p


def _delete_output_for_segment(
    *,
    output_root: Path,
    project_root: Path,
    sport: str,
    event: str,
    kind: str,
    seg_id: str,
    apply: bool,
) -> tuple[int, int]:
    deleted_outputs = 0
    deleted_mot = 0

    out_json = output_root / sport / event / kind / f"{seg_id}.json"
    mot_dir = output_root / sport / event / kind / "mot"

    mot_refs: set[str] = set()
    if out_json.exists():
        try:
            mot_refs = _extract_mot_refs(_load_json(out_json))
        except Exception:
            mot_refs = set()

        if apply:
            try:
                out_json.unlink()
                deleted_outputs += 1
            except OSError:
                pass
        else:
            deleted_outputs += 1

    if mot_dir.exists():
        fallback = set(str(p) for p in mot_dir.glob(f"{seg_id}_*.txt"))
        for ref in mot_refs:
            fallback.add(str(_resolve_mot_path(project_root, ref)))

        for ref in sorted(fallback):
            mot_path = Path(ref)
            if apply:
                try:
                    if mot_path.exists():
                        mot_path.unlink()
                        deleted_mot += 1
                except OSError:
                    pass
            else:
                if mot_path.exists():
                    deleted_mot += 1

    return deleted_outputs, deleted_mot


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fix corrupted sport/event fields in JSONs under data/Dataset by inferring {sport}/{event} from the path."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Dataset root (default: data/Dataset)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/output"),
        help="Output root to delete affected outputs (default: data/output)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root to resolve relative mot_file refs (default: cwd)",
    )
    parser.add_argument(
        "--no-delete-output",
        action="store_true",
        help="Do not delete output JSON/MOT files when fixing clips/frames metadata",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write fixes in-place (default: dry-run)",
    )
    parser.add_argument(
        "--print",
        dest="print_paths",
        action="store_true",
        help="Print each file that would be changed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N JSONs (debug)",
    )

    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    output_root: Path = args.output_root
    project_root: Path = args.project_root
    delete_output = not bool(args.no_delete_output)

    if not dataset_root.exists() or not dataset_root.is_dir():
        print(
            f"dataset_root not found or not a directory: {dataset_root}",
            file=sys.stderr,
        )
        return 2

    paths = [p for p in _iter_dataset_jsons(dataset_root) if p.is_file()]
    if args.limit is not None:
        paths = paths[: max(0, args.limit)]

    changed: list[FixResult] = []
    errors = 0
    outputs_marked = 0
    mot_marked = 0

    for path in paths:
        se = _infer_sport_event(dataset_root, path)
        if se is None:
            continue
        sport, event = se

        try:
            data = _load_json(path)
        except Exception:
            errors += 1
            continue

        kind = path.parent.name
        declared_sport, declared_event = _get_declared_sport_event(data)

        new_data, fixed_origin, fixed_top = fix_one(data, sport=sport, event=event)
        if not (fixed_origin or fixed_top):
            continue

        if kind in ("frames", "clips") and delete_output:
            seg_id = _get_segment_id(new_data, default=path.stem)
            candidates: set[tuple[str, str]] = {(sport, event)}
            if declared_sport is not None and declared_event is not None:
                candidates.add((declared_sport, declared_event))

            for s, e in sorted(candidates):
                o, m = _delete_output_for_segment(
                    output_root=output_root,
                    project_root=project_root,
                    sport=s,
                    event=e,
                    kind=kind,
                    seg_id=seg_id,
                    apply=bool(args.apply),
                )
                outputs_marked += o
                mot_marked += m

        changed.append(
            FixResult(
                path=path,
                expected_sport=sport,
                expected_event=event,
                fixed_origin=fixed_origin,
                fixed_top_level=fixed_top,
            )
        )

        if args.print_paths:
            print(path)

        if args.apply:
            _write_json(path, new_data)

    print("dataset_root:", dataset_root)
    print("output_root :", output_root)
    print("apply       :", bool(args.apply))
    print("delete_output:", bool(delete_output))
    print("scanned     :", len(paths))
    print("errors      :", errors)
    print("changed     :", len(changed))

    if delete_output:
        label_outputs = "outputs_deleted" if args.apply else "outputs_would_delete"
        label_mot = "mot_deleted" if args.apply else "mot_would_delete"
        print(f"{label_outputs}: {outputs_marked}")
        print(f"{label_mot}: {mot_marked}")

    if changed and not args.apply:
        print("Run with --apply to write fixes.")

    return 1 if changed or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
