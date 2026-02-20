from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from scripts.annotate_ai_coach import annotate_ai_coach_batch, normalize_ai_coach_response


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_clip_metadata(
    *,
    dataset_root: Path,
    sport: str,
    event: str,
    clip_id: str,
    tasks: list[str],
) -> None:
    event_dir = dataset_root / sport / event / "clips"
    event_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        event_dir / f"{clip_id}.json",
        {
            "id": clip_id,
            "origin": {"sport": sport, "event": event},
            "info": {
                "original_starting_frame": 0,
                "total_frames": 120,
                "fps": 10.0,
            },
            "tasks_to_annotate": tasks,
        },
    )
    (event_dir / f"{clip_id}.mp4").write_bytes(b"fake-video")


class FakeGeminiClient:
    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.uploaded: list[Path] = []
        self.cleaned: list[Any] = []
        self.annotate_calls = 0
        self.prompts: list[str] = []

    def upload_video(self, video_path: Path) -> Any:
        self.uploaded.append(video_path)
        return {"uri": str(video_path)}

    def annotate_video(self, video_file: Any, prompt: str) -> Any:
        self.annotate_calls += 1
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("No fake response configured")
        index = min(self.annotate_calls - 1, len(self._responses) - 1)
        return self._responses[index]

    def cleanup_file(self, file_obj: Any) -> None:
        self.cleaned.append(file_obj)


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
        return {"qa_pairs": [{"question": "Q", "answer": "A"}]}

    def cleanup_file(self, file_obj: Any) -> None:
        return None


def test_normalize_ai_coach_response_variants() -> None:
    parsed = normalize_ai_coach_response(
        {
            "qa_pairs": [
                {"question": "Q1", "answer": "A1"},
                {"Q": "Q2", "A": "A2"},
            ]
        }
    )
    assert parsed == [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]

    assert normalize_ai_coach_response({"question": "Q", "answer": "A"}) == [
        {"question": "Q", "answer": "A"}
    ]
    assert normalize_ai_coach_response([{"question": "Q", "answer": "A"}]) == [
        {"question": "Q", "answer": "A"}
    ]
    assert normalize_ai_coach_response({"qa_pairs": ["Q: What happened? A: Foot fault"]}) == [
        {"question": "What happened?", "answer": "Foot fault"}
    ]

    with pytest.raises(ValueError):
        normalize_ai_coach_response({"qa_pairs": []})


def test_annotate_ai_coach_batch_writes_output_and_skips_existing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    output_root = tmp_path / "output"
    _write_clip_metadata(
        dataset_root=dataset_root,
        sport="SportA",
        event="EventA",
        clip_id="1",
        tasks=["AI_Coach"],
    )
    _write_clip_metadata(
        dataset_root=dataset_root,
        sport="SportA",
        event="EventA",
        clip_id="2",
        tasks=["ScoreboardSingle"],
    )

    fake_client = FakeGeminiClient(
        responses=[
            {
                "qa_pairs": [
                    {
                        "question": "What mistake did the athlete make?",
                        "answer": "The athlete stepped on the line.",
                    },
                    {
                        "question": "extra",
                        "answer": "extra",
                    }
                ]
            }
        ]
    )
    prompt_template = "total={total_frames}; fps={fps}; {language_instruction}"

    stats_first = annotate_ai_coach_batch(
        dataset_root=dataset_root,
        output_root=output_root,
        gemini_client=fake_client,
        prompt_template=prompt_template,
    )
    assert stats_first.scanned_jsons == 2
    assert stats_first.matched_ai_coach == 1
    assert stats_first.annotated == 1
    assert stats_first.skipped_existing == 0
    assert stats_first.failed == 0
    assert fake_client.annotate_calls == 1

    output_path = output_root / "SportA" / "EventA" / "clips" / "1.json"
    assert output_path.is_file()
    output_data = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_data["id"] == "1"
    assert output_data["origin"] == {"sport": "SportA", "event": "EventA"}
    assert len(output_data["annotations"]) == 1
    assert output_data["annotations"][0]["task_L2"] == "AI_Coach"
    assert len(output_data["annotations"][0]["qa_pairs"]) == 1
    assert output_data["annotations"][0]["qa_pairs"][0]["answer"] == "The athlete stepped on the line."

    stats_second = annotate_ai_coach_batch(
        dataset_root=dataset_root,
        output_root=output_root,
        gemini_client=fake_client,
        prompt_template=prompt_template,
    )
    assert stats_second.scanned_jsons == 2
    assert stats_second.matched_ai_coach == 1
    assert stats_second.annotated == 0
    assert stats_second.skipped_existing == 1
    assert stats_second.failed == 0
    assert fake_client.annotate_calls == 1


def test_overwrite_replaces_existing_ai_coach_but_keeps_other_annotations(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    output_root = tmp_path / "output"
    _write_clip_metadata(
        dataset_root=dataset_root,
        sport="SportB",
        event="EventB",
        clip_id="7",
        tasks=["AI_Coach"],
    )

    output_path = output_root / "SportB" / "EventB" / "clips" / "7.json"
    _write_json(
        output_path,
        {
            "id": "7",
            "origin": {"sport": "SportB", "event": "EventB"},
            "annotations": [
                {
                    "annotation_id": "1",
                    "task_L1": "Understanding",
                    "task_L2": "AI_Coach",
                    "qa_pairs": [{"question": "Old Q", "answer": "Old A"}],
                },
                {
                    "annotation_id": "2",
                    "task_L1": "Understanding",
                    "task_L2": "ScoreboardSingle",
                    "question": "Existing",
                    "answer": ["keep"],
                },
            ],
        },
    )

    fake_client = FakeGeminiClient(
        responses=[
            {
                "qa_pairs": [
                    {"question": "New Q", "answer": "New A"},
                ]
            }
        ]
    )
    prompt_template = "fps={fps}; {language_instruction}"

    stats = annotate_ai_coach_batch(
        dataset_root=dataset_root,
        output_root=output_root,
        gemini_client=fake_client,
        prompt_template=prompt_template,
        overwrite=True,
    )
    assert stats.annotated == 1
    assert stats.failed == 0

    updated = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(updated["annotations"]) == 2
    assert {ann["task_L2"] for ann in updated["annotations"]} == {
        "AI_Coach",
        "ScoreboardSingle",
    }
    ai_coach_ann = [ann for ann in updated["annotations"] if ann["task_L2"] == "AI_Coach"][0]
    assert ai_coach_ann["qa_pairs"] == [{"question": "New Q", "answer": "New A"}]
    assert {ann["annotation_id"] for ann in updated["annotations"]} == {"1", "2"}


def test_parallel_annotation_with_factory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    output_root = tmp_path / "output"
    _write_clip_metadata(
        dataset_root=dataset_root,
        sport="SportP",
        event="EventP",
        clip_id="1",
        tasks=["AI_Coach"],
    )
    _write_clip_metadata(
        dataset_root=dataset_root,
        sport="SportP",
        event="EventP",
        clip_id="2",
        tasks=["AI_Coach"],
    )
    _write_clip_metadata(
        dataset_root=dataset_root,
        sport="SportP",
        event="EventP",
        clip_id="3",
        tasks=["AI_Coach"],
    )

    FactoryGeminiClient.reset()
    stats = annotate_ai_coach_batch(
        dataset_root=dataset_root,
        output_root=output_root,
        gemini_client_factory=FactoryGeminiClient,
        prompt_template="total={total_frames}; fps={fps}",
        num_workers=2,
        progress=False,
    )
    assert stats.scanned_jsons == 3
    assert stats.matched_ai_coach == 3
    assert stats.annotated == 3
    assert stats.failed == 0
    assert FactoryGeminiClient.call_count == 3

    for clip_id in ("1", "2", "3"):
        out_path = output_root / "SportP" / "EventP" / "clips" / f"{clip_id}.json"
        assert out_path.is_file()
