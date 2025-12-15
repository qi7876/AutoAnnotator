#!/usr/bin/env python
"""Generate segment metadata JSON files.

This script automatically generates or updates segment_metadata.json files
from video files, extracting metadata and organizing it according to the
new simplified schema.

Usage:
    python scripts/generate_segment_metadata.py <segment_video_path> [options]

Examples:
    # Basic usage (auto-detect all information)
    python scripts/generate_segment_metadata.py Dataset/Archery/Men's_Individual/segments/3.mp4

    # Specify segment ID explicitly
    python scripts/generate_segment_metadata.py segment.mp4 --segment-id 5

    # Specify output file
    python scripts/generate_segment_metadata.py segment.mp4 -o output.json
"""

import json
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_annotator.utils import VideoUtils


def extract_sport_event_from_path(video_path: Path) -> tuple[str, str]:
    """Extract sport and event information from path.

    Args:
        video_path: Video file path

    Returns:
        (sport, event) tuple
    """
    parts = list(video_path.parts)

    # Try to extract info from path
    # Example: Dataset/Archery/Men's_Individual/segments/3.mp4
    sport = "Unknown"
    event = "Unknown"

    # Find Dataset directory index
    dataset_idx = None
    for i, part in enumerate(parts):
        if "Dataset" in part:
            dataset_idx = i
            break

    if dataset_idx is not None and dataset_idx + 1 < len(parts):
        # Sport is the first directory after Dataset
        sport = parts[dataset_idx + 1]

        # Event is the second directory
        if dataset_idx + 2 < len(parts):
            event_part = parts[dataset_idx + 2]
            # Skip segments directory
            if event_part != "segments" and not event_part.endswith(".mp4"):
                event = event_part

    return sport, event


def extract_segment_id_from_path(video_path: Path) -> int:
    """Extract segment ID from video path.

    Args:
        video_path: Video file path

    Returns:
        segment_id as integer
    """
    # Example: 3.mp4 -> 3
    # Example: segment_5.mp4 -> 5
    stem = video_path.stem

    # Try to extract number
    import re
    numbers = re.findall(r'\d+', stem)

    if numbers:
        return int(numbers[-1])  # Take last number

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate segment metadata JSON file"
    )
    parser.add_argument(
        "segment_video",
        type=str,
        help="Segment video file path"
    )
    parser.add_argument(
        "--segment-id",
        type=int,
        help="Segment ID (will auto-detect from filename if not specified)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path (default: segment_metadata_<id>.json)"
    )
    parser.add_argument(
        "--sport",
        type=str,
        help="Sport type (optional, will auto-detect from path)"
    )
    parser.add_argument(
        "--event",
        type=str,
        help="Event name (optional, will auto-detect from path)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame in original video (default: 0)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["ScoreboardSingle"],
        help="Tasks to annotate (default: ScoreboardSingle)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Additional description"
    )

    args = parser.parse_args()

    # Parse path
    segment_path = Path(args.segment_video).resolve()

    if not segment_path.exists():
        print(f"Error: Video file not found: {segment_path}")
        return 1

    print("=" * 60)
    print("Generate Segment Metadata")
    print("=" * 60)
    print(f"Segment video: {segment_path}")

    # 1. Get video info
    print("\nReading video information...")
    try:
        video_info = VideoUtils.get_video_info(segment_path)
        print(f"  FPS: {video_info['fps']}")
        print(f"  Total frames: {video_info['total_frames']}")
        print(f"  Resolution: {video_info['resolution']}")
        print(f"  Duration: {video_info['duration_sec']} seconds")
    except Exception as e:
        print(f"Error: Cannot read video info: {e}")
        return 1

    # 2. Extract sport and event
    sport = args.sport
    event = args.event

    if not sport or not event:
        print("\nExtracting sport and event from path...")
        auto_sport, auto_event = extract_sport_event_from_path(segment_path)
        if not sport:
            sport = auto_sport
        if not event:
            event = auto_event
        print(f"  Sport: {sport}")
        print(f"  Event: {event}")

    # 3. Extract or use segment_id
    if args.segment_id is not None:
        segment_id = args.segment_id
    else:
        segment_id = extract_segment_id_from_path(segment_path)

    print(f"\nSegment ID: {segment_id}")

    # 4. Generate JSON
    metadata = {
        "segment_id": segment_id,
        "original_video": {
            "sport": sport,
            "event": event
        },
        "segment_info": {
            "start_frame_in_original": args.start_frame,
            "total_frames": video_info["total_frames"],
            "fps": float(video_info["fps"]),
            "duration_sec": video_info["duration_sec"],
            "resolution": video_info["resolution"]
        },
        "tasks_to_annotate": args.tasks,
        "additional_info": {
            "description": args.description if args.description else f"Extracted from {segment_path.name}"
        }
    }

    # 5. Save JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"segment_metadata_{segment_id}.json")

    print(f"\nSaving to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print(f"\nGenerated JSON file: {output_path}")
    print("\nContent preview:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    print("\nNote: Video paths will be automatically constructed as:")
    print(f"  Segment: Dataset/{sport}/{event}/segments/{segment_id}.mp4")
    print(f"  Original: Dataset/{sport}/{event}/1.mp4")

    return 0


if __name__ == "__main__":
    sys.exit(main())
