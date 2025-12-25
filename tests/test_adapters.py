"""Tests for the input adapter module."""

import json
import pytest
from pathlib import Path
from auto_annotator.adapters import InputAdapter, ClipMetadata, ClipInfo, OriginInfo


class TestClipMetadataCreation:
    """Test creating ClipMetadata from dict."""

    def test_create_clip_metadata(self):
        """Test creating video clip metadata."""
        data = {
            "id": "1",
            "origin": {
                "sport": "3x3_Basketball",
                "event": "Men"
            },
            "info": {
                "original_starting_frame": 6520,
                "total_frames": 70,
                "fps": 10.0
            },
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)
        assert metadata.id == "1"
        assert metadata.origin.sport == "3x3_Basketball"
        assert metadata.origin.event == "Men"
        assert metadata.info.original_starting_frame == 6520
        assert metadata.info.total_frames == 70
        assert metadata.info.fps == 10.0
        assert metadata.info.is_clip()
        assert not metadata.info.is_single_frame()
        assert metadata.tasks_to_annotate == ["UCE"]

    def test_create_singleframe_metadata(self):
        """Test creating single frame metadata."""
        data = {
            "id": "1",
            "origin": {
                "sport": "Archery",
                "event": "Men's_Individual"
            },
            "info": {
                "original_starting_frame": 7462,
                "total_frames": 1,
                "fps": 10.0
            },
            "tasks_to_annotate": ["ScoreboardSingle"]
        }

        metadata = InputAdapter.create_from_dict(data)
        assert metadata.id == "1"
        assert metadata.origin.sport == "Archery"
        assert metadata.origin.event == "Men's_Individual"
        assert metadata.info.original_starting_frame == 7462
        assert metadata.info.total_frames == 1
        assert metadata.info.is_single_frame()
        assert not metadata.info.is_clip()


class TestClipMetadataTyping:
    """Test type detection methods."""

    def test_is_single_frame(self):
        """Test single frame detection."""
        data = {
            "id": "5",
            "origin": {"sport": "Archery", "event": "Men's_Individual"},
            "info": {"original_starting_frame": 100, "total_frames": 1, "fps": 10.0},
            "tasks_to_annotate": ["ScoreboardSingle"]
        }

        metadata = InputAdapter.create_from_dict(data)
        assert metadata.info.is_single_frame()
        assert not metadata.info.is_clip()

    def test_is_clip(self):
        """Test video clip detection."""
        data = {
            "id": "2",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)
        assert metadata.info.is_clip()
        assert not metadata.info.is_single_frame()


class TestPathConstruction:
    """Test path construction methods."""

    def test_clip_video_path(self):
        """Test constructing path to clip video."""
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)
        expected_path = Path("data/Dataset/3x3_Basketball/Men/clips/1.mp4")
        assert metadata.get_video_path() == expected_path

        # Test with custom dataset root
        custom_root = Path("/custom/dataset")
        expected_path = custom_root / "3x3_Basketball/Men/clips/1.mp4"
        assert metadata.get_video_path(custom_root) == expected_path

    def test_singleframe_image_path(self):
        """Test constructing path to single frame image."""
        data = {
            "id": "1",
            "origin": {"sport": "Archery", "event": "Men's_Individual"},
            "info": {"original_starting_frame": 7462, "total_frames": 1, "fps": 10.0},
            "tasks_to_annotate": ["ScoreboardSingle"]
        }

        metadata = InputAdapter.create_from_dict(data)
        expected_path = Path("data/Dataset/Archery/Men's_Individual/frames/1.jpg")
        assert metadata.get_video_path() == expected_path

        # Test JSON path
        expected_json_path = Path("data/Dataset/Archery/Men's_Individual/frames/1.json")
        assert metadata.get_json_path() == expected_json_path

    def test_original_video_path(self):
        """Test constructing path to original video."""
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)

        # Default to video_id = "1"
        expected_path = Path("data/Dataset/3x3_Basketball/Men/1.mp4")
        assert metadata.get_original_video_path() == expected_path

        # Test with specific video_id
        expected_path = Path("data/Dataset/3x3_Basketball/Men/2.mp4")
        assert metadata.get_original_video_path(video_id="2") == expected_path

    def test_custom_dataset_root(self):
        """Test all paths with custom dataset root."""
        data = {
            "id": "3",
            "origin": {"sport": "Archery", "event": "Men's_Individual"},
            "info": {"original_starting_frame": 1000, "total_frames": 50, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)
        custom_root = Path("/my/custom/dataset")

        # Test clip video path
        expected_video = custom_root / "Archery/Men's_Individual/clips/3.mp4"
        assert metadata.get_video_path(custom_root) == expected_video

        # Test JSON path
        expected_json = custom_root / "Archery/Men's_Individual/clips/3.json"
        assert metadata.get_json_path(custom_root) == expected_json

        # Test original video path
        expected_original = custom_root / "Archery/Men's_Individual/1.mp4"
        assert metadata.get_original_video_path(custom_root) == expected_original


class TestTaskMethods:
    """Test task-related methods."""

    def test_has_task(self):
        """Test checking if a task is present."""
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": ["UCE", "ScoreboardMultiple"]
        }

        metadata = InputAdapter.create_from_dict(data)
        assert metadata.has_task("UCE")
        assert metadata.has_task("ScoreboardMultiple")
        assert not metadata.has_task("Object_Tracking")


class TestMetadataValidation:
    """Test metadata validation."""

    def test_validate_valid_clip(self):
        """Test validation of valid clip metadata."""
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)
        is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
        assert is_valid
        assert error is None

    def test_validate_invalid_starting_frame(self):
        """Test validation of negative starting frame."""
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": -10, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)
        is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
        assert not is_valid
        assert "original_starting_frame must be non-negative" in error

    def test_validate_invalid_total_frames(self):
        """Test validation of invalid total_frames."""
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 0, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)
        is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
        assert not is_valid
        assert "total_frames must be positive" in error

    def test_validate_no_tasks(self):
        """Test validation of metadata with no tasks."""
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": []
        }

        metadata = InputAdapter.create_from_dict(data)
        is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
        assert not is_valid
        assert "No tasks specified" in error


class TestOriginInfo:
    """Test OriginInfo class."""

    def test_origin_video_paths(self):
        """Test origin video path construction."""
        origin = OriginInfo(sport="3x3_Basketball", event="Men")

        # Test default video_id
        expected_path = Path("data/Dataset/3x3_Basketball/Men/1.mp4")
        assert origin.get_video_path() == expected_path

        # Test custom video_id
        expected_path = Path("data/Dataset/3x3_Basketball/Men/2.mp4")
        assert origin.get_video_path(video_id="2") == expected_path

        # Test JSON path
        expected_json = Path("data/Dataset/3x3_Basketball/Men/1.json")
        assert origin.get_json_path() == expected_json

        # Test metainfo path
        expected_metainfo = Path("data/Dataset/3x3_Basketball/Men/metainfo.json")
        assert origin.get_metainfo_path() == expected_metainfo

    def test_origin_custom_root(self):
        """Test origin paths with custom dataset root."""
        origin = OriginInfo(sport="Archery", event="Men's_Individual")
        custom_root = Path("/custom/dataset")

        expected_video = custom_root / "Archery/Men's_Individual/1.mp4"
        assert origin.get_video_path(custom_root) == expected_video

        expected_json = custom_root / "Archery/Men's_Individual/1.json"
        assert origin.get_json_path(custom_root) == expected_json

        expected_metainfo = custom_root / "Archery/Men's_Individual/metainfo.json"
        assert origin.get_metainfo_path(custom_root) == expected_metainfo


class TestJSONLoading:
    """Test JSON loading functionality."""

    def test_load_from_dict(self):
        """Test loading metadata from dict."""
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        metadata = InputAdapter.create_from_dict(data)
        assert isinstance(metadata, ClipMetadata)
        assert metadata.id == "1"
        assert metadata.origin.sport == "3x3_Basketball"

    def test_load_from_json_file(self, tmp_path):
        """Test loading metadata from JSON file."""
        # Create a temporary JSON file
        json_file = tmp_path / "1.json"
        data = {
            "id": "1",
            "origin": {"sport": "3x3_Basketball", "event": "Men"},
            "info": {"original_starting_frame": 6520, "total_frames": 70, "fps": 10.0},
            "tasks_to_annotate": ["UCE"]
        }

        with open(json_file, "w") as f:
            json.dump(data, f)

        # Load the metadata
        metadata = InputAdapter.load_from_json(json_file)
        assert isinstance(metadata, ClipMetadata)
        assert metadata.id == "1"
        assert metadata.origin.sport == "3x3_Basketball"

    def test_load_from_json_file_not_found(self):
        """Test loading metadata from non-existent file."""
        with pytest.raises(FileNotFoundError):
            InputAdapter.load_from_json(Path("/nonexistent/file.json"))
