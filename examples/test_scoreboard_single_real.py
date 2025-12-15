#!/usr/bin/env python
"""Real test script for ScoreboardSingleAnnotator.

This script demonstrates how to use the ScoreboardSingleAnnotator
with real segment metadata.

Usage:
    python examples/test_scoreboard_single_real.py <segment_metadata.json>

Example:
    python examples/test_scoreboard_single_real.py segment_metadata_3.json
"""

import json
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auto_annotator.adapters import InputAdapter
from auto_annotator.annotators import GeminiClient, TaskAnnotatorFactory
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.utils import PromptLoader
from auto_annotator.config import get_config


def main():
    """Main function: Real annotation using ScoreboardSingleAnnotator."""

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python examples/test_scoreboard_single_real.py <segment_metadata.json>")
        print("\nExample segment_metadata.json format:")
        print(json.dumps({
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
                "description": "Extracted single frame at time 746.200s."
            }
        }, indent=2, ensure_ascii=False))
        sys.exit(1)

    segment_metadata_path = Path(sys.argv[1])

    if not segment_metadata_path.exists():
        logger.error(f"File not found: {segment_metadata_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("ScoreboardSingle Real Annotation Test")
    logger.info("=" * 60)

    try:
        # 1. Load configuration
        logger.info("Loading configuration...")
        config = get_config()
        logger.info(f"Using Gemini model: {config.gemini.model}")
        logger.info(f"Dataset root: {config.dataset_root}")

        # 2. Load segment metadata
        logger.info(f"Loading segment metadata: {segment_metadata_path}")
        segment_metadata = InputAdapter.load_from_json(segment_metadata_path)
        logger.info(f"Segment ID: {segment_metadata.segment_id}")
        logger.info(f"Sport: {segment_metadata.original_video.sport}")
        logger.info(f"Event: {segment_metadata.original_video.event}")

        # Construct video path
        video_path = segment_metadata.get_video_path(config.dataset_root)
        logger.info(f"Video path: {video_path}")
        logger.info(f"Tasks: {segment_metadata.tasks_to_annotate}")

        # Validate metadata
        is_valid, error = InputAdapter.validate_metadata(
            segment_metadata,
            dataset_root=config.dataset_root
        )
        if not is_valid:
            logger.error(f"Segment metadata validation failed: {error}")
            sys.exit(1)

        # Check if ScoreboardSingle task is included
        if "ScoreboardSingle" not in segment_metadata.tasks_to_annotate:
            logger.warning("Segment metadata doesn't include ScoreboardSingle task")
            logger.info("Adding ScoreboardSingle to task list...")
            segment_metadata.tasks_to_annotate = ["ScoreboardSingle"]

        # 3. Initialize components
        logger.info("Initializing components...")
        gemini_client = GeminiClient()
        prompt_loader = PromptLoader()
        bbox_annotator = BBoxAnnotator(gemini_client)
        tracker = ObjectTracker()

        # 4. Create ScoreboardSingleAnnotator
        logger.info("Creating ScoreboardSingleAnnotator...")
        annotator = TaskAnnotatorFactory.create_annotator(
            task_name="ScoreboardSingle",
            gemini_client=gemini_client,
            prompt_loader=prompt_loader,
            bbox_annotator=bbox_annotator,
            tracker=tracker
        )

        logger.info(f"Task name: {annotator.get_task_name()}")
        logger.info(f"Task category: {annotator.get_task_l1()}")

        # 5. Execute annotation
        logger.info("=" * 60)
        logger.info("Starting annotation...")
        logger.info("=" * 60)

        annotation_result = annotator.annotate(
            segment_metadata,
            dataset_root=config.dataset_root
        )

        # 6. Display results
        logger.info("=" * 60)
        logger.info("Annotation Results")
        logger.info("=" * 60)

        print("\n" + "=" * 60)
        print("Final Annotation Result (JSON format):")
        print("=" * 60)
        print(json.dumps(annotation_result, indent=2, ensure_ascii=False))

        if 'bounding_box' in annotation_result:
            bbox = annotation_result['bounding_box']
            if isinstance(bbox, list) and len(bbox) == 4:
                print(f"\nBounding box coordinates [xtl, ytl, xbr, ybr]:")
                print(f"  {bbox}")
            else:
                print(f"\nBounding box: {bbox}")

        debug_info = annotation_result.get("_debug", {})
        frame_path = debug_info.get("frame_path")
        if frame_path:
            print(f"\nExtracted frame path: {frame_path}")

        raw_annotation = getattr(gemini_client, "last_annotation_raw", None)
        if raw_annotation:
            print("\n" + "=" * 60)
            print("Gemini 2.5 Flash Raw Response:")
            print("=" * 60)
            print(raw_annotation)

        raw_grounding = getattr(gemini_client, "last_grounding_raw", None)
        if raw_grounding:
            print("\n" + "=" * 60)
            print("Robotics ER 1.5 Grounding Raw Response:")
            print("=" * 60)
            print(raw_grounding)

        # 7. Validate results
        logger.info("\nValidating annotation results...")
        is_valid, error = annotator.validate_annotation(annotation_result)
        if is_valid:
            logger.info("✓ Annotation results validated successfully")
        else:
            logger.warning(f"⚠ Annotation validation warning: {error}")

        print("\n" + "=" * 60)
        print("Annotation complete!")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
