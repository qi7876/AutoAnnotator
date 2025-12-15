#!/usr/bin/env python3
"""Test script to verify the updated input adapter with unified metadata format."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test importing the updated adapter
from src.auto_annotator.adapters.input_adapter import (
    InputAdapter,
    SegmentMetadata,
    OriginalVideoInfo,
    SegmentInfo
)


def test_segment_metadata():
    """Test video segment metadata."""
    print("\n=== Testing Video Segment Metadata ===")

    segment_data = {
        "segment_id": "1_split_7_start_000652",
        "original_video": {
            "sport": "3x3_Basketball",
            "event": "Men"
        },
        "segment_info": {
            "start_frame_in_original": 6520,
            "total_frames": 70,
            "fps": 10.0,
            "duration_sec": 7.0,
            "resolution": [1920, 1080]
        },
        "tasks_to_annotate": ["UCE"],
        "additional_info": {
            "description": "Test segment"
        }
    }

    metadata = InputAdapter.create_from_dict(segment_data)
    print(f"✓ Segment metadata created: {metadata.segment_id}")
    print(f"  Is segment: {metadata.segment_info.is_segment()}")
    print(f"  Is single frame: {metadata.segment_info.is_single_frame()}")

    # Test path construction
    video_path = metadata.get_video_path()
    json_path = metadata.get_json_path()
    original_path = metadata.get_original_video_path()

    print(f"  Segment video path: {video_path}")
    print(f"  Segment JSON path: {json_path}")
    print(f"  Original video path: {original_path}")

    # Verify path format
    expected_video = "Dataset/3x3_Basketball/Men/segment_dir/1_split_7_start_000652.mp4"
    assert str(video_path) == expected_video, f"Expected {expected_video}, got {video_path}"
    print("✓ Segment path format is correct")

    # Test video_id extraction
    parsed = metadata._parse_segment_id()
    assert parsed["video_id"] == 1, f"Expected video_id=1, got {parsed['video_id']}"
    assert parsed["split_num"] == 7, f"Expected split_num=7, got {parsed['split_num']}"
    assert parsed["start_frame"] == 652, f"Expected start_frame=652, got {parsed['start_frame']}"
    print(f"✓ Parsed segment_id correctly: video_id={parsed['video_id']}, split_num={parsed['split_num']}, start_frame={parsed['start_frame']}")

    # Test validation (without file existence check)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert is_valid, f"Validation failed: {error}"
    print("✓ Segment validation passed")


def test_singleframe_metadata():
    """Test single frame metadata."""
    print("\n=== Testing Single Frame Metadata ===")

    frame_data = {
        "segment_id": 5,
        "original_video": {
            "sport": "Archery",
            "event": "Men's_Individual"
        },
        "segment_info": {
            "start_frame_in_original": 7462,
            "total_frames": 1,
            "fps": 10.0,
            "duration_sec": 0.1,
            "resolution": [1920, 1080]
        },
        "tasks_to_annotate": ["ScoreboardSingle"],
        "additional_info": {
            "description": "Single frame at 746.2s"
        }
    }

    metadata = InputAdapter.create_from_dict(frame_data)
    print(f"✓ Frame metadata created: {metadata.segment_id}")
    print(f"  Is segment: {metadata.segment_info.is_segment()}")
    print(f"  Is single frame: {metadata.segment_info.is_single_frame()}")

    # Test path construction
    image_path = metadata.get_video_path()  # Returns image path for singleframe
    json_path = metadata.get_json_path()
    original_path = metadata.get_original_video_path()

    print(f"  Frame image path: {image_path}")
    print(f"  Frame JSON path: {json_path}")
    print(f"  Original video path: {original_path}")

    # Verify path format
    expected_image = "Dataset/Archery/Men's_Individual/singleframes_dir/5.jpg"
    assert str(image_path) == expected_image, f"Expected {expected_image}, got {image_path}"
    print("✓ Frame path format is correct")

    # Test validation (without file existence check)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert is_valid, f"Validation failed: {error}"
    print("✓ Frame validation passed")


def test_original_video_info():
    """Test original video info methods."""
    print("\n=== Testing Original Video Info ===")

    video_info = OriginalVideoInfo(
        sport="Archery",
        event="Women's_Team"
    )

    # Test with default video_id
    video_path = video_info.get_video_path(video_id=1)
    json_path = video_info.get_json_path(video_id=1)
    metainfo_path = video_info.get_metainfo_path()

    print(f"  Video path (id=1): {video_path}")
    print(f"  JSON path (id=1): {json_path}")
    print(f"  Metainfo path: {metainfo_path}")

    assert str(video_path) == "Dataset/Archery/Women's_Team/1.mp4"
    assert str(json_path) == "Dataset/Archery/Women's_Team/1.json"
    assert str(metainfo_path) == "Dataset/Archery/Women's_Team/metainfo.json"

    # Test with different video_id
    video_path_2 = video_info.get_video_path(video_id=7)
    assert str(video_path_2) == "Dataset/Archery/Women's_Team/7.mp4"

    print("✓ Original video paths are correct")


def test_segment_id_parsing():
    """Test different segment_id formats."""
    print("\n=== Testing Segment ID Parsing ===")

    # Test format: {video_id}_split_{split_num}_start_{start_frame}
    test_cases = [
        ("1_split_7_start_000652", {"video_id": 1, "split_num": 7, "start_frame": 652}),
        ("3_split_2_start_001234", {"video_id": 3, "split_num": 2, "start_frame": 1234}),
        ("1_frame_3401", {"video_id": 1, "frame_num": 3401}),
        (5, {"video_id": 1}),  # Integer segment_id
    ]

    for segment_id, expected in test_cases:
        metadata = SegmentMetadata(
            segment_id=segment_id,
            original_video=OriginalVideoInfo(sport="Test", event="Event"),
            segment_info=SegmentInfo(
                start_frame_in_original=0,
                total_frames=1 if isinstance(segment_id, int) else 100,
                fps=10.0,
                duration_sec=0.1 if isinstance(segment_id, int) else 10.0,
                resolution=[1920, 1080]
            ),
            tasks_to_annotate=["test"]
        )
        parsed = metadata._parse_segment_id()
        assert parsed["video_id"] == expected["video_id"], f"Failed for {segment_id}: {parsed}"
        print(f"✓ Parsed '{segment_id}' -> video_id={parsed['video_id']}")


def test_validation_errors():
    """Test that validation catches errors correctly."""
    print("\n=== Testing Validation Error Detection ===")

    # Test negative start_frame
    bad_segment = {
        "segment_id": "1_split_1_start_000100",
        "original_video": {
            "sport": "Test",
            "event": "Event"
        },
        "segment_info": {
            "start_frame_in_original": -1,  # Invalid
            "total_frames": 100,
            "fps": 10.0,
            "duration_sec": 10.0,
            "resolution": [1920, 1080]
        },
        "tasks_to_annotate": ["test"]
    }
    metadata = InputAdapter.create_from_dict(bad_segment)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert not is_valid, "Should have failed with negative start_frame"
    print(f"✓ Correctly caught error: {error}")

    # Test invalid duration
    bad_duration = {
        "segment_id": "1_split_1_start_000100",
        "original_video": {
            "sport": "Test",
            "event": "Event"
        },
        "segment_info": {
            "start_frame_in_original": 100,
            "total_frames": 100,
            "fps": 10.0,
            "duration_sec": 5.0,  # Should be 10.0
            "resolution": [1920, 1080]
        },
        "tasks_to_annotate": ["test"]
    }
    metadata = InputAdapter.create_from_dict(bad_duration)
    is_valid, error = InputAdapter.validate_metadata(metadata, check_file_existence=False)
    assert not is_valid, "Should have failed with inconsistent duration"
    print(f"✓ Correctly caught error: {error}")


def test_json_serialization():
    """Test saving and loading from JSON."""
    print("\n=== Testing JSON Serialization ===")

    # Create test metadata
    segment_data = {
        "segment_id": "2_split_3_start_001500",
        "original_video": {
            "sport": "Basketball",
            "event": "Men"
        },
        "segment_info": {
            "start_frame_in_original": 15000,
            "total_frames": 50,
            "fps": 10.0,
            "duration_sec": 5.0,
            "resolution": [1920, 1080]
        },
        "tasks_to_annotate": ["UCE"],
        "additional_info": {
            "description": "Test segment"
        }
    }

    # Save to JSON
    test_json_path = Path("test_segment_metadata.json")
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(segment_data, f, indent=2)
    print(f"✓ Saved metadata to {test_json_path}")

    # Load from JSON
    loaded_metadata = InputAdapter.load_from_json(test_json_path)
    print(f"✓ Loaded metadata: {loaded_metadata.segment_id}")

    # Verify data matches
    assert loaded_metadata.segment_id == segment_data["segment_id"]
    assert loaded_metadata.original_video.sport == segment_data["original_video"]["sport"]
    assert loaded_metadata.segment_info.start_frame_in_original == segment_data["segment_info"]["start_frame_in_original"]
    print("✓ Loaded data matches original")

    # Clean up
    test_json_path.unlink()
    print(f"✓ Cleaned up {test_json_path}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Unified Metadata Format")
    print("=" * 60)

    try:
        test_segment_metadata()
        test_singleframe_metadata()
        test_original_video_info()
        test_segment_id_parsing()
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
