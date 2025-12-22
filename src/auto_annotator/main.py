"""Main entry point for AutoAnnotator."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from .adapters import InputAdapter, ClipMetadata
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
    segment_metadata: ClipMetadata,
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
    logger.info(f"Processing clip: {segment_metadata.id}")

    # Validate segment metadata
    is_valid, error = InputAdapter.validate_metadata(
        segment_metadata,
        dataset_root=dataset_root
    )
    if not is_valid:
        logger.error(f"Invalid clip metadata: {error}")
        raise ValueError(f"Invalid clip metadata: {error}")

    output_path = output_dir / f"{segment_metadata.id}.json"
    existing_output = None
    completed_tasks: set[str] = set()

    if output_path.exists():
        try:
            existing_output = JSONUtils.load_json(output_path)
            for ann in existing_output.get("annotations", []):
                task_name = ann.get("task_L2")
                if task_name:
                    completed_tasks.add(task_name)
        except Exception as e:
            logger.warning(
                f"Failed to load existing output {output_path}: {e}. "
                "Re-annotating all tasks."
            )

    tasks_to_run = [
        task_name
        for task_name in segment_metadata.tasks_to_annotate
        if task_name not in completed_tasks
    ]

    if not tasks_to_run:
        logger.info(f"All tasks already annotated for: {segment_metadata.id}")
        return output_path

    # Collect annotations for missing tasks
    annotations = []

    for task_name in tasks_to_run:
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

    if existing_output:
        output_data = JSONUtils.merge_annotations(existing_output, annotations)
    else:
        output_data = {
            "id": segment_metadata.id,
            "origin": {
                "sport": segment_metadata.origin.sport,
                "event": segment_metadata.origin.event,
            },
            "annotations": annotations
        }

    JSONUtils.save_json(output_data, output_path)
    logger.info(f"Saved {len(annotations)} annotations to {output_path}")

    return output_path


def _load_segment_metadata(
    segment_paths: Iterable[Path]
) -> List[Tuple[Path, ClipMetadata]]:
    """Load segment metadata files and return pairs of (path, metadata)."""
    logger = logging.getLogger(__name__)
    loaded: List[Tuple[Path, ClipMetadata]] = []

    for segment_path in segment_paths:
        try:
            logger.info(f"Loading segment metadata: {segment_path}")
            metadata = InputAdapter.load_from_json(segment_path)
            loaded.append((segment_path, metadata))
        except Exception as e:
            logger.error(f"Failed to load {segment_path}: {e}", exc_info=True)
            continue

    return loaded


def _prune_orphan_outputs(
    output_dir: Path,
    valid_clip_ids: set[str]
):
    """Remove output files whose source metadata no longer exists."""
    logger = logging.getLogger(__name__)
    if not output_dir.exists():
        return

    for output_path in output_dir.glob("*.json"):
        try:
            data = JSONUtils.load_json(output_path)
        except Exception as e:
            logger.warning(f"Skipping unreadable output file: {output_path} ({e})")
            continue

        if not isinstance(data, dict) or "annotations" not in data:
            continue

        clip_id = data.get("id", output_path.stem)
        if clip_id not in valid_clip_ids:
            try:
                output_path.unlink()
                logger.info(f"Removed orphan output: {output_path}")
            except Exception as e:
                logger.error(f"Failed to remove {output_path}: {e}", exc_info=True)


def process_segments_batch(
    segment_paths: List[Path],
    output_dir: Path,
    prune_orphans: bool = False
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

    # Load metadata upfront to support incremental behavior
    loaded_segments = _load_segment_metadata(segment_paths)
    valid_clip_ids = {metadata.id for _, metadata in loaded_segments}

    if prune_orphans and loaded_segments:
        _prune_orphan_outputs(output_dir, valid_clip_ids)

    # Process each segment
    successful = 0
    failed = 0

    for segment_path, segment_metadata in loaded_segments:
        try:
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
            logger.info(f"Successfully processed clip: {segment_metadata.id}")

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
        process_segments_batch(
            segment_paths,
            output_dir,
            prune_orphans=segment_path.is_dir()
        )

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
