"""Main entry point for AutoAnnotator."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .adapters import InputAdapter, SegmentMetadata
from .annotators import GeminiClient, TaskAnnotatorFactory
from .annotators.bbox_annotator import BBoxAnnotator
from .annotators.tracker import ObjectTracker
from .config import get_config, get_config_manager
from .utils import JSONUtils, PromptLoader


def setup_logging():
    """Setup logging configuration."""
    config = get_config()

    # Create logs directory
    log_file = Path(config.project_root) / config.logging.file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("AutoAnnotator started")
    logger.info("="*60)


def process_segment(
    segment_metadata: SegmentMetadata,
    gemini_client: GeminiClient,
    prompt_loader: PromptLoader,
    bbox_annotator: BBoxAnnotator,
    tracker: ObjectTracker,
    output_dir: Path,
    dataset_root: Path
) -> Path:
    """
    Process a single segment.

    Args:
        segment_metadata: Segment metadata
        gemini_client: Gemini API client
        prompt_loader: Prompt loader
        bbox_annotator: Bounding box annotator
        tracker: Object tracker
        output_dir: Output directory for results
        dataset_root: Root directory of the dataset

    Returns:
        Path to output JSON file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing segment: {segment_metadata.segment_id}")

    # Validate segment metadata
    is_valid, error = InputAdapter.validate_metadata(
        segment_metadata,
        dataset_root=dataset_root
    )
    if not is_valid:
        logger.error(f"Invalid segment metadata: {error}")
        raise ValueError(f"Invalid segment metadata: {error}")

    # Collect annotations for all tasks
    annotations = []

    for task_name in segment_metadata.tasks_to_annotate:
        logger.info(f"Annotating task: {task_name}")

        try:
            # Create annotator for this task
            annotator = TaskAnnotatorFactory.create_annotator(
                task_name=task_name,
                gemini_client=gemini_client,
                prompt_loader=prompt_loader,
                bbox_annotator=bbox_annotator,
                tracker=tracker
            )

            # Perform annotation
            annotation = annotator.annotate(segment_metadata, dataset_root=dataset_root)

            # Validate annotation
            is_valid, error = annotator.validate_annotation(annotation)
            if not is_valid:
                logger.warning(f"Invalid annotation for {task_name}: {error}")
                continue

            annotations.append(annotation)
            logger.info(f"Successfully annotated {task_name}")

        except NotImplementedError as e:
            logger.warning(f"Task {task_name} requires unimplemented features: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to annotate {task_name}: {e}", exc_info=True)
            continue

    # Save annotations to temp output
    output_path = output_dir / f"{segment_metadata.segment_id}.json"

    output_data = {
        "segment_id": segment_metadata.segment_id,
        "original_video": {
            "sport": segment_metadata.original_video.sport,
            "event": segment_metadata.original_video.event,
        },
        "annotations": annotations
    }

    JSONUtils.save_json(output_data, output_path)
    logger.info(f"Saved {len(annotations)} annotations to {output_path}")

    return output_path


def process_segments_batch(
    segment_paths: List[Path],
    output_dir: Path
):
    """
    Process a batch of segments.

    Args:
        segment_paths: List of paths to segment metadata JSON files
        output_dir: Output directory for results
    """
    logger = logging.getLogger(__name__)
    config = get_config()

    # Initialize components
    gemini_client = GeminiClient()
    prompt_loader = PromptLoader()
    bbox_annotator = BBoxAnnotator()
    tracker = ObjectTracker(backend=config.tasks.tracking.get("tracker_backend", "local"))

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each segment
    successful = 0
    failed = 0

    for segment_path in segment_paths:
        try:
            logger.info(f"Loading segment metadata: {segment_path}")

            # Load segment metadata
            segment_metadata = InputAdapter.load_from_json(segment_path)

            # Process segment
            output_path = process_segment(
                segment_metadata=segment_metadata,
                gemini_client=gemini_client,
                prompt_loader=prompt_loader,
                bbox_annotator=bbox_annotator,
                tracker=tracker,
                output_dir=output_dir,
                dataset_root=config.dataset_root
            )

            successful += 1
            logger.info(f"Successfully processed segment: {segment_metadata.segment_id}")

        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {segment_path}: {e}", exc_info=True)
            continue

    # Summary
    logger.info("="*60)
    logger.info("Processing complete")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {successful + failed}")
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AutoAnnotator - AI-powered video annotation system"
    )

    parser.add_argument(
        "segment_path",
        type=str,
        help="Path to segment metadata JSON file or directory containing multiple files"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output directory for annotation results (default: from config)",
        default=None
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)

    try:
        # Get output directory
        config = get_config()
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path(config.project_root) / config.output.temp_dir

        # Process segments
        segment_path = Path(args.segment_path)

        if segment_path.is_file():
            # Single segment file
            segment_paths = [segment_path]
        elif segment_path.is_dir():
            # Directory containing multiple segment files
            segment_paths = list(segment_path.glob("*.json"))
            logger.info(f"Found {len(segment_paths)} segment files in {segment_path}")
        else:
            raise FileNotFoundError(f"Path not found: {segment_path}")

        if not segment_paths:
            logger.error("No segment files found")
            return 1

        # Process batch
        process_segments_batch(segment_paths, output_dir)

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
