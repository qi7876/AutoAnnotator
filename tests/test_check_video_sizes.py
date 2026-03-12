from __future__ import annotations

import json
from pathlib import Path

from scripts.check_video_sizes import collect_video_sizes, main


def _write_bytes(path: Path, size_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size_bytes)


def test_collect_video_sizes_ignores_non_video_files(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_bytes(root / "a.mp4", 1024)
    _write_bytes(root / "nested" / "b.MOV", 2048)
    _write_bytes(root / "nested" / "note.txt", 4096)

    files = collect_video_sizes(root)

    assert [item.path.relative_to(root).as_posix() for item in files] == [
        "a.mp4",
        "nested/b.MOV",
    ]
    assert [item.size_bytes for item in files] == [1024, 2048]


def test_check_video_sizes_main_outputs_summary_and_json(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "data"
    json_out = tmp_path / "report" / "video_sizes.json"
    _write_bytes(root / "clips" / "small.mp4", 1024)
    _write_bytes(root / "clips" / "large.mkv", 4096)

    exit_code = main(
        [
            "--root",
            str(root),
            "--top",
            "2",
            "--json-out",
            str(json_out),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Video file size summary" in captured.out
    assert "Files: 2" in captured.out
    assert "Top 2 largest files" in captured.out
    assert ".mkv: 1 files, 4.0 KiB" in captured.out

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["file_count"] == 2
    assert payload["total_bytes"] == 5120
    assert payload["max_file"] == {
        "path": "clips/large.mkv",
        "size_bytes": 4096,
    }


def test_check_video_sizes_main_returns_1_when_file_exceeds_threshold(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "data"
    _write_bytes(root / "ok.mp4", 1024)
    _write_bytes(root / "too_large.mp4", 4096)

    exit_code = main(
        [
            "--root",
            str(root),
            "--top",
            "0",
            "--max-size-mb",
            "0.003",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Files over 3.1 KiB: 1" in captured.out
    assert "too_large.mp4" in captured.out
