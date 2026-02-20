from __future__ import annotations

import json
from pathlib import Path

from scripts.expand_task_abbreviations import normalize_task_abbreviations


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_normalize_task_abbreviations_dry_run_and_apply(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    path1 = dataset_root / "SportA" / "EventA" / "clips" / "1.json"
    path2 = dataset_root / "SportA" / "EventA" / "frames" / "2.json"
    path3 = dataset_root / "SportA" / "EventB" / "clips" / "3.json"

    _write_json(
        path1,
        {
            "id": "a",
            "tasks_to_annotate": [
                "UCE",
                "UCA",
                "USM",
                "STG",
                "USS",
                "UOS",
                "OSR",
                "RSI",
                "POT",
                "ScoreboardSingle",
            ],
        },
    )
    _write_json(path2, {"id": "b", "task_to_annotate": ["UCA", "RSI", "Continuous_Actions_Caption"]})
    _write_json(path3, {"id": "c", "tasks_to_annotate": ["ScoreboardSingle", "Spatial_Temporal_Grounding"]})

    json_paths = sorted(dataset_root.rglob("*.json"))
    changes, replaced_counter, removed_counter, invalid_json = normalize_task_abbreviations(
        json_paths=json_paths,
        apply=False,
    )

    assert len(changes) == 2
    assert replaced_counter["UCE"] == 1
    assert replaced_counter["UCA"] == 2
    assert replaced_counter["USM"] == 1
    assert replaced_counter["STG"] == 1
    assert replaced_counter["USS"] == 1
    assert replaced_counter["UOS"] == 1
    assert replaced_counter["OSR"] == 1
    assert removed_counter["RSI"] == 2
    assert removed_counter["POT"] == 1
    assert invalid_json == 0

    dry_run_data = json.loads(path1.read_text(encoding="utf-8"))
    assert dry_run_data["tasks_to_annotate"][0] == "UCE"

    normalize_task_abbreviations(json_paths=json_paths, apply=True)

    updated1 = json.loads(path1.read_text(encoding="utf-8"))
    assert updated1["tasks_to_annotate"] == [
        "Continuous_Events_Caption",
        "Continuous_Actions_Caption",
        "ScoreboardMultiple",
        "Spatial_Temporal_Grounding",
        "ScoreboardSingle",
        "Objects_Spatial_Relationships",
    ]

    updated2 = json.loads(path2.read_text(encoding="utf-8"))
    assert updated2["task_to_annotate"] == ["Continuous_Actions_Caption"]

    unchanged3 = json.loads(path3.read_text(encoding="utf-8"))
    assert unchanged3["tasks_to_annotate"] == ["ScoreboardSingle", "Spatial_Temporal_Grounding"]
