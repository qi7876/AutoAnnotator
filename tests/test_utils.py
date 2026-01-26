"""Tests for utility modules."""

import pytest
from pathlib import Path

from auto_annotator.utils import JSONUtils, PromptLoader, VideoUtils


class TestJSONUtils:
    """Tests for JSONUtils."""

    def test_merge_annotations(self):
        """Test merging annotations."""
        base_json = {
            "sport": "Basketball",
            "event": "Men",
            "video_id": "1",
            "video_metadata": {},
            "annotations": [
                {"annotation_id": "1", "task_L1": "Understanding"}
            ]
        }

        new_annotations = [
            {"task_L1": "Understanding", "task_L2": "ScoreboardSingle"},
            {"task_L1": "Perception", "task_L2": "Object_Tracking"}
        ]

        result = JSONUtils.merge_annotations(base_json, new_annotations)

        assert len(result["annotations"]) == 3
        assert result["annotations"][1]["annotation_id"] == "2"
        assert result["annotations"][2]["annotation_id"] == "3"

    def test_validate_annotation_json_valid(self):
        """Test validating valid annotation JSON."""
        data = {
            "sport": "Basketball",
            "event": "Men",
            "video_id": "1",
            "video_metadata": {
                "duration_sec": 10.0,
                "fps": 10,
                "total_frames": 100,
                "resolution": [1920, 1080]
            },
            "annotations": []
        }

        is_valid, error = JSONUtils.validate_annotation_json(data)
        assert is_valid
        assert error == ""

    def test_validate_annotation_json_missing_field(self):
        """Test validating annotation JSON with missing field."""
        data = {
            "sport": "Basketball",
            "event": "Men",
            # Missing video_id
            "video_metadata": {},
            "annotations": []
        }

        is_valid, error = JSONUtils.validate_annotation_json(data)
        assert not is_valid
        assert "video_id" in error

    def test_filter_annotations_by_task(self):
        """Test filtering annotations by task."""
        data = {
            "annotations": [
                {"annotation_id": "1", "task_L1": "Understanding", "task_L2": "ScoreboardSingle"},
                {"annotation_id": "2", "task_L1": "Understanding", "task_L2": "ScoreboardMultiple"},
                {"annotation_id": "3", "task_L1": "Perception", "task_L2": "Object_Tracking"},
            ]
        }

        # Filter by L1
        result = JSONUtils.filter_annotations_by_task(data, task_l1="Understanding")
        assert len(result) == 2

        # Filter by L2
        result = JSONUtils.filter_annotations_by_task(data, task_l2="Object_Tracking")
        assert len(result) == 1

        # Filter by both
        result = JSONUtils.filter_annotations_by_task(
            data, task_l1="Understanding", task_l2="ScoreboardSingle"
        )
        assert len(result) == 1

    def test_get_annotation_ids(self):
        """Test getting annotation IDs."""
        data = {
            "annotations": [
                {"annotation_id": "1"},
                {"annotation_id": "2"},
                {"annotation_id": "3"}
            ]
        }

        ids = JSONUtils.get_annotation_ids(data)
        assert ids == ["1", "2", "3"]


class TestPromptLoader:
    """Tests for PromptLoader."""

    def test_list_available_tasks(self):
        """Test listing available tasks."""
        loader = PromptLoader()
        tasks = loader.list_available_tasks()

        assert "ScoreboardSingle" in tasks
        assert "Object_Tracking" in tasks
        assert len(tasks) == 7

    def test_get_required_variables(self):
        """Test getting required variables for a task."""
        loader = PromptLoader()
        variables = loader.get_required_variables("ScoreboardSingle")

        assert len(variables) > 0
        assert "total_frames" in variables
        assert "max_frame" in variables

    def test_validate_prompt_files(self):
        """Test validating prompt files exist."""
        loader = PromptLoader()
        status = loader.validate_prompt_files()

        for task, exists in status.items():
            assert exists, f"Prompt file for {task} doesn't exist"


class TestVideoUtils:
    """Tests for VideoUtils."""

    def test_frames_to_seconds(self):
        """Test converting frames to seconds."""
        assert VideoUtils.frames_to_seconds(30, 10) == 3.0
        assert VideoUtils.frames_to_seconds(100, 10) == 10.0

    def test_seconds_to_frames(self):
        """Test converting seconds to frames."""
        assert VideoUtils.seconds_to_frames(3.0, 10) == 30
        assert VideoUtils.seconds_to_frames(10.0, 10) == 100

    def test_frames_to_seconds_invalid_fps(self):
        """Test frames to seconds with invalid FPS."""
        with pytest.raises(ValueError):
            VideoUtils.frames_to_seconds(30, 0)

        with pytest.raises(ValueError):
            VideoUtils.frames_to_seconds(30, -1)
