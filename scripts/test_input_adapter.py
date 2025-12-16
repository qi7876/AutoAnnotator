#!/usr/bin/env python3
"""Test script to verify the updated input adapter with simplified metadata format."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test importing the updated adapter
from src.auto_annotator.adapters.input_adapter import (
    InputAdapter,
    ClipMetadata,
    OriginInfo,
    ClipInfo
)


def test_clip_metadata():
    """Test video clip metadata."""
    print("\n=== Testing Video Clip Metadata ===")

    clip_data = {
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

    metadata = InputAdapter.create_from_dict(clip_data)
    print(f"✓ Clip metadata created: {metadata.id}")
    print(f"  Is clip: {metadata.info.is_clip()}")
    print(f"  Is single frame: {metadata.info.is_single_frame()}")

    # Test path construction
    video_path = metadata.get_video_path()
    json_path = metadata.get_json_path()
    original_path = metadata.get_original_video_path()

    print(f"  Clip video path: {video_path}")
    print(f"  Clip JSON path: {json_path}")
    print(f"  Original video path: {original_path}")

    # Verify path format
    expected_video = "Dataset/3x3_Basketball/Men/clips/1.mp4"
    assert str(video_path) == expected_video, f"Expected {expected_video}, got {video_path}"
    print("✓ Clip path format is correct")

    # Test validation (without file existence check)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert is_valid, f"Validation failed: {error}"
    print("✓ Clip validation passed")


def test_singleframe_metadata():
    """Test single frame metadata."""
    print("\n=== Testing Single Frame Metadata ===")

    frame_data = {
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

    metadata = InputAdapter.create_from_dict(frame_data)
    print(f"✓ Frame metadata created: {metadata.id}")
    print(f"  Is clip: {metadata.info.is_clip()}")
    print(f"  Is single frame: {metadata.info.is_single_frame()}")

    # Test path construction
    image_path = metadata.get_video_path()  # Returns image path for singleframe
    json_path = metadata.get_json_path()
    original_path = metadata.get_original_video_path()

    print(f"  Frame image path: {image_path}")
    print(f"  Frame JSON path: {json_path}")
    print(f"  Original video path: {original_path}")

    # Verify path format
    expected_image = "Dataset/Archery/Men's_Individual/frames/1.jpg"
    assert str(image_path) == expected_image, f"Expected {expected_image}, got {image_path}"
    print("✓ Frame path format is correct")

    # Test validation (without file existence check)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert is_valid, f"Validation failed: {error}"
    print("✓ Frame validation passed")


def test_origin_info():
    """Test origin info methods."""
    print("\n=== Testing Origin Info ===")

    origin_info = OriginInfo(
        sport="Archery",
        event="Women's_Team"
    )

    # Test with default video_id
    video_path = origin_info.get_video_path(video_id="1")
    json_path = origin_info.get_json_path(video_id="1")
    metainfo_path = origin_info.get_metainfo_path()

    print(f"  Video path (id=1): {video_path}")
    print(f"  JSON path (id=1): {json_path}")
    print(f"  Metainfo path: {metainfo_path}")

    assert str(video_path) == "Dataset/Archery/Women's_Team/1.mp4"
    assert str(json_path) == "Dataset/Archery/Women's_Team/1.json"
    assert str(metainfo_path) == "Dataset/Archery/Women's_Team/metainfo.json"

    # Test with different video_id
    video_path_2 = origin_info.get_video_path(video_id="7")
    assert str(video_path_2) == "Dataset/Archery/Women's_Team/7.mp4"

    print("✓ Origin video paths are correct")


def test_validation_errors():
    """Test that validation catches errors correctly."""
    print("\n=== Testing Validation Error Detection ===")

    # Test negative original_starting_frame
    bad_clip = {
        "id": "1",
        "origin": {
            "sport": "Test",
            "event": "Event"
        },
        "info": {
            "original_starting_frame": -1,  # Invalid
            "total_frames": 100,
            "fps": 10.0
        },
        "tasks_to_annotate": ["test"]
    }
    metadata = InputAdapter.create_from_dict(bad_clip)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert not is_valid, "Should have failed with negative original_starting_frame"
    print(f"✓ Correctly caught error: {error}")

    # Test zero total_frames
    bad_frames = {
        "id": "1",
        "origin": {
            "sport": "Test",
            "event": "Event"
        },
        "info": {
            "original_starting_frame": 100,
            "total_frames": 0,  # Invalid
            "fps": 10.0
        },
        "tasks_to_annotate": ["test"]
    }
    metadata = InputAdapter.create_from_dict(bad_frames)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert not is_valid, "Should have failed with zero total_frames"
    print(f"✓ Correctly caught error: {error}")

    # Test no tasks
    no_tasks = {
        "id": "1",
        "origin": {
            "sport": "Test",
            "event": "Event"
        },
        "info": {
            "original_starting_frame": 100,
            "total_frames": 50,
            "fps": 10.0
        },
        "tasks_to_annotate": []  # Invalid
    }
    metadata = InputAdapter.create_from_dict(no_tasks)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert not is_valid, "Should have failed with no tasks"
    print(f"✓ Correctly caught error: {error}")


def test_json_serialization():
    """Test saving and loading from JSON."""
    print("\n=== Testing JSON Serialization ===")

    # Create test metadata
    clip_data = {
        "id": "2",
        "origin": {
            "sport": "Basketball",
            "event": "Men"
        },
        "info": {
            "original_starting_frame": 15000,
            "total_frames": 50,
            "fps": 10.0
        },
        "tasks_to_annotate": ["UCE"]
    }

    # Save to JSON
    test_json_path = Path("test_clip_metadata.json")
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(clip_data, f, indent=2)
    print(f"✓ Saved metadata to {test_json_path}")

    # Load from JSON
    loaded_metadata = InputAdapter.load_from_json(test_json_path)
    print(f"✓ Loaded metadata: {loaded_metadata.id}")

    # Verify data matches
    assert loaded_metadata.id == clip_data["id"]
    assert loaded_metadata.origin.sport == clip_data["origin"]["sport"]
    assert loaded_metadata.info.original_starting_frame == clip_data["info"]["original_starting_frame"]
    print("✓ Loaded data matches original")

    # Clean up
    test_json_path.unlink()
    print(f"✓ Cleaned up {test_json_path}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Simplified Metadata Format")
    print("=" * 60)

    try:
        test_clip_metadata()
        test_singleframe_metadata()
        test_origin_info()
        test_validation_errors()
        test_json_serialization()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
