"""Tests for input adapters."""

import pytest
from pathlib import Path

from auto_annotator.adapters import InputAdapter, SegmentMetadata


def test_segment_metadata_creation():
    """Test creating SegmentMetadata from dict."""
    data = {
        "segment_id": "test_001",
        "original_video": {
            "path": "/path/to/video.mp4",
            "json_path": "/path/to/video.json",
            "sport": "Basketball",
            "event": "Men",
            "video_id": "1"
        },
        "segment_info": {
            "path": "/path/to/segment.mp4",
            "start_frame_in_original": 100,
            "total_frames": 50,
            "fps": 10,
            "duration_sec": 5.0,
            "resolution": [1920, 1080]
        },
        "tasks_to_annotate": ["ScoreboardSingle"],
        "additional_info": {}
    }

    metadata = InputAdapter.create_from_dict(data)
    assert metadata.segment_id == "test_001"
    assert metadata.original_video.sport == "Basketball"
    assert metadata.segment_info.fps == 10


def test_segment_metadata_has_task():
    """Test checking if segment has specific task."""
    data = {
        "segment_id": "test_001",
        "original_video": {
            "path": "/path/to/video.mp4",
            "json_path": "/path/to/video.json",
            "sport": "Basketball",
            "event": "Men",
            "video_id": "1"
        },
        "segment_info": {
            "path": "/path/to/segment.mp4",
            "start_frame_in_original": 100,
            "total_frames": 50,
            "fps": 10,
            "duration_sec": 5.0
        },
        "tasks_to_annotate": ["ScoreboardSingle", "Object_Tracking"]
    }

    metadata = InputAdapter.create_from_dict(data)
    assert metadata.has_task("ScoreboardSingle")
    assert metadata.has_task("Object_Tracking")
    assert not metadata.has_task("NonExistentTask")


def test_validate_metadata_invalid_frames():
    """Test validating metadata with invalid frame numbers."""
    data = {
        "segment_id": "test_001",
        "original_video": {
            "path": Path(__file__).parent / "fixtures" / "video.mp4",
            "json_path": Path(__file__).parent / "fixtures" / "video.json",
            "sport": "Basketball",
            "event": "Men",
            "video_id": "1"
        },
        "segment_info": {
            "path": Path(__file__).parent / "fixtures" / "segment.mp4",
            "start_frame_in_original": -1,  # Invalid
            "total_frames": 50,
            "fps": 10,
            "duration_sec": 5.0
        },
        "tasks_to_annotate": ["ScoreboardSingle"]
    }

    metadata = InputAdapter.create_from_dict(data)
    is_valid, error = InputAdapter.validate_metadata(metadata)
    assert not is_valid
    assert "non-negative" in error


def test_validate_metadata_no_tasks():
    """Test validating metadata with no tasks."""
    data = {
        "segment_id": "test_001",
        "original_video": {
            "path": Path(__file__).parent / "fixtures" / "video.mp4",
            "json_path": Path(__file__).parent / "fixtures" / "video.json",
            "sport": "Basketball",
            "event": "Men",
            "video_id": "1"
        },
        "segment_info": {
            "path": Path(__file__).parent / "fixtures" / "segment.mp4",
            "start_frame_in_original": 0,
            "total_frames": 50,
            "fps": 10,
            "duration_sec": 5.0
        },
        "tasks_to_annotate": []  # No tasks
    }

    metadata = InputAdapter.create_from_dict(data)
    is_valid, error = InputAdapter.validate_metadata(metadata)
    assert not is_valid
    assert "No tasks" in error
