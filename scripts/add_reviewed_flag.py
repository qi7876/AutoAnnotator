#!/usr/bin/env python3
"""Add a reviewed flag to all annotations in data/output JSONs if missing."""

from __future__ import annotations

import json
from pathlib import Path


def add_reviewed_flag(json_path: Path) -> bool:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        print(f"skip (unreadable): {json_path}")
        return False

    anns = data.get("annotations")
    if not isinstance(anns, list):
        print(f"skip (annotations not list): {json_path}")
        return False

    updated = False
    for ann in anns:
        if isinstance(ann, dict) and "reviewed" not in ann:
            ann["reviewed"] = False
            updated = True

    if updated:
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"updated: {json_path}")
    return updated


def main() -> int:
    output_root = Path("data/output")
    if not output_root.exists():
        print(f"output root not found: {output_root}")
        return 1

    json_files = sorted(output_root.rglob("*.json"))
    total = len(json_files)
    updated = 0
    for idx, path in enumerate(json_files, 1):
        updated += 1 if add_reviewed_flag(path) else 0
    print(f"processed={total} updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
