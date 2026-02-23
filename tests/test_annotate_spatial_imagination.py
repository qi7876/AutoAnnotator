from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from scripts.annotate_spatial_imagination import (
    annotate_spatial_imagination_batch,
    normalize_spatial_imagination_response,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_spatial_imagination_clip(
    *,
    dataset_root: Path,
    sport: str,
    event: str,
    clip_id: str,
    tasks: list[str],
    source_task_l2: str = "Spatial_Temporal_Grounding",
    source_ref: str = "The athlete in red on the right side",
) -> None:
    clip_dir = dataset_root / sport / event / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        clip_dir / f"{clip_id}.json",
        {
            "id": clip_id,
            "origin": {"sport": sport, "event": event, "video_id": "1"},
            "info": {
                "original_starting_frame": 200,
                "total_frames": 24,
                "fps": 10.0,
                "Q_window_frame": [200, 223],
            },
            "tasks_to_annotate": tasks,
            "source_annotation": {
                "source_annotation_id": "7",
                "task_L1": "Understanding",
                "task_L2": source_task_l2,
                "annotation": {
                    "annotation_id": "7",
                    "task_L1": "Understanding",
                    "task_L2": source_task_l2,
                    "query": source_ref,
                    "A_window_frame": [200, 223],
                    "tracking_bboxes": "./dataset/x/mot/fake.txt",
                },
            },
        },
    )
    (clip_dir / f"{clip_id}.mp4").write_bytes(b"fake-video")


class FakeGeminiClient:
    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.annotate_calls = 0
        self.uploaded: list[Path] = []
        self.prompts: list[str] = []

    def upload_video(self, video_path: Path) -> Any:
        self.uploaded.append(video_path)
        return {"uri": str(video_path)}

    def annotate_video(self, video_file: Any, prompt: str) -> Any:
        self.annotate_calls += 1
        self.prompts.append(prompt)
        index = min(self.annotate_calls - 1, len(self._responses) - 1)
        return self._responses[index]

    def cleanup_file(self, file_obj: Any) -> None:
        return None


class FactoryGeminiClient:
    call_lock = threading.Lock()
    call_count = 0

    @classmethod
    def reset(cls) -> None:
        with cls.call_lock:
            cls.call_count = 0

    def upload_video(self, video_path: Path) -> Any:
        return {"uri": str(video_path)}

    def annotate_video(self, video_file: Any, prompt: str) -> Any:
        with self.call_lock:
            type(self).call_count += 1
        return {"question": "Q?", "answer": "A."}

    def cleanup_file(self, file_obj: Any) -> None:
        return None


def test_normalize_spatial_imagination_response_variants() -> None:
    assert normalize_spatial_imagination_response(
        {"question": "Q1", "answer": "A1"}
    ) == ("Q1", "A1")
    assert normalize_spatial_imagination_response(
        {"qa": {"question": "Q2", "answer": "A2"}}
    ) == ("Q2", "A2")
    assert normalize_spatial_imagination_response(
        {"qa_pairs": [{"question": "Q3", "answer": "A3"}]}
    ) == ("Q3", "A3")
    assert normalize_spatial_imagination_response(
        [{"question": "Q4", "answer": "A4"}]
    ) == ("Q4", "A4")
    assert normalize_spatial_imagination_response(
        "Q: What is the path? A: The path bends left."
    ) == ("What is the path?", "The path bends left.")

    with pytest.raises(ValueError):
        normalize_spatial_imagination_response({"foo": "bar"})


def test_spatial_imagination_batch_writes_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    output_root = tmp_path / "output"
    _write_spatial_imagination_clip(
        dataset_root=dataset_root,
        sport="SportA",
        event="EventA",
        clip_id="1",
        tasks=["Spatial_Imagination"],
        source_ref="The Serbian player in white under the basket",
    )
    _write_spatial_imagination_clip(
        dataset_root=dataset_root,
        sport="SportA",
        event="EventA",
        clip_id="2",
        tasks=["AI_Coach"],
    )

    fake_client = FakeGeminiClient(
        responses=[
            {
                "question": "From a top-down view, where does the target move?",
                "answer": "The player moves diagonally from right baseline to center paint.",
            }
        ]
    )

    stats = annotate_spatial_imagination_batch(
        dataset_root=dataset_root,
        output_root=output_root,
        gemini_client=fake_client,
        prompt_template=(
            "source_task={source_task_l2}\n"
            "source={source_object_reference}\n"
            "frames={total_frames}"
        ),
        progress=False,
    )
    assert stats.scanned_jsons == 2
    assert stats.matched_spatial_imagination == 1
    assert stats.annotated == 1
    assert stats.failed == 0
    assert fake_client.annotate_calls == 1
    assert "The Serbian player in white under the basket" in fake_client.prompts[0]

    out_path = output_root / "SportA" / "EventA" / "clips" / "1.json"
    assert out_path.is_file()
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["origin"]["video_id"] == "1"
    assert len(out["annotations"]) == 1
    ann = out["annotations"][0]
    assert ann["task_L2"] == "Spatial_Imagination"
    assert "top-down view" in ann["question"]
    assert "diagonally" in ann["answer"]


def test_spatial_imagination_skip_existing_and_overwrite(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    output_root = tmp_path / "output"
    _write_spatial_imagination_clip(
        dataset_root=dataset_root,
        sport="SportB",
        event="EventB",
        clip_id="8",
        tasks=["Spatial_Imagination"],
    )
    out_path = output_root / "SportB" / "EventB" / "clips" / "8.json"
    _write_json(
        out_path,
        {
            "id": "8",
            "origin": {"sport": "SportB", "event": "EventB", "video_id": "1"},
            "annotations": [
                {
                    "annotation_id": "1",
                    "task_L1": "Understanding",
                    "task_L2": "Spatial_Imagination",
                    "question": "Old Q",
                    "answer": "Old A",
                },
                {
                    "annotation_id": "2",
                    "task_L1": "Understanding",
                    "task_L2": "ScoreboardSingle",
                    "question": "Keep",
                    "answer": ["keep"],
                },
            ],
        },
    )

    fake_client = FakeGeminiClient(
        responses=[{"question": "New Q", "answer": "New A"}]
    )
    stats_skip = annotate_spatial_imagination_batch(
        dataset_root=dataset_root,
        output_root=output_root,
        gemini_client=fake_client,
        prompt_template="source={source_object_reference}",
        progress=False,
    )
    assert stats_skip.annotated == 0
    assert stats_skip.skipped_existing == 1
    assert fake_client.annotate_calls == 0

    stats_overwrite = annotate_spatial_imagination_batch(
        dataset_root=dataset_root,
        output_root=output_root,
        gemini_client=fake_client,
        prompt_template="source={source_object_reference}",
        overwrite=True,
        progress=False,
    )
    assert stats_overwrite.annotated == 1
    updated = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(updated["annotations"]) == 2
    task_names = {ann["task_L2"] for ann in updated["annotations"]}
    assert task_names == {"Spatial_Imagination", "ScoreboardSingle"}
    spatial_ann = [ann for ann in updated["annotations"] if ann["task_L2"] == "Spatial_Imagination"][0]
    assert spatial_ann["question"] == "New Q"
    assert spatial_ann["answer"] == "New A"


def test_spatial_imagination_parallel_with_factory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    output_root = tmp_path / "output"
    for clip_id in ("1", "2", "3"):
        _write_spatial_imagination_clip(
            dataset_root=dataset_root,
            sport="SportP",
            event="EventP",
            clip_id=clip_id,
            tasks=["Spatial_Imagination"],
        )

    FactoryGeminiClient.reset()
    stats = annotate_spatial_imagination_batch(
        dataset_root=dataset_root,
        output_root=output_root,
        gemini_client_factory=FactoryGeminiClient,
        prompt_template="source={source_object_reference}",
        num_workers=2,
        progress=False,
    )
    assert stats.annotated == 3
    assert stats.failed == 0
    assert FactoryGeminiClient.call_count == 3
