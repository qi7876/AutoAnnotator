"""Main entry point for AutoAnnotator."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .adapters import ClipMetadata, InputAdapter
from .annotators import GeminiClient, TaskAnnotatorFactory
from .annotators.bbox_annotator import BBoxAnnotator
from .annotators.tracker import ObjectTracker
from .config import get_config
from .utils import JSONUtils, PromptLoader


class ColorFormatter(logging.Formatter):
    """Add ANSI colors to console logs."""

    COLORS = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, use_color: bool):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self.use_color:
            return message
        color = self.COLORS.get(record.levelno)
        if not color:
            return message
        return f"{color}{message}{self.RESET}"


def setup_logging():
    """Setup logging configuration."""
    config = get_config()

    # Create logs directory
    log_file = Path(config.project_root) / config.logging.file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Logging to: {log_file}")

    # Configure logging
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, config.logging.level))

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter(config.logging.format))

    use_color = sys.stdout.isatty()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColorFormatter(config.logging.format, use_color))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("AutoAnnotator started")
    logger.info("=" * 60)


def process_segment(
    segment_metadata: ClipMetadata,
    gemini_client: GeminiClient,
    prompt_loader: PromptLoader,
    bbox_annotator: BBoxAnnotator,
    tracker: ObjectTracker,
    output_dir: Path,
    dataset_root: Path,
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
    segment_type = "frame" if segment_metadata.info.is_single_frame() else "clip"
    logger.info(f"Processing {segment_type}: {segment_metadata.id}")

    # Validate segment metadata
    is_valid, error = InputAdapter.validate_metadata(
        segment_metadata, dataset_root=dataset_root
    )
    if not is_valid:
        logger.error(f"Invalid clip metadata: {error}")
        raise ValueError(f"Invalid clip metadata: {error}")

    output_path = output_dir / f"{segment_metadata.id}.json"
    existing_output = None
    completed_tasks: set[str] = set()
    existing_annotations: list[dict] = []

    if output_path.exists():
        try:
            existing_output = JSONUtils.load_json(output_path)
            existing_annotations = existing_output.get("annotations", [])
            if not isinstance(existing_annotations, list):
                existing_annotations = []
        except Exception as e:
            logger.warning(
                f"Failed to load existing output {output_path}: {e}. "
                "Re-annotating all tasks."
            )

    available_tasks = set(TaskAnnotatorFactory.get_available_tasks())
    requested_tasks = list(segment_metadata.tasks_to_annotate)
    unknown_tasks = sorted(set(requested_tasks) - available_tasks)
    if unknown_tasks:
        logger.warning(
            "Skipping unknown tasks for %s %s: %s",
            segment_type,
            segment_metadata.id,
            ", ".join(unknown_tasks),
        )

    annotator_cache: dict[str, Any] = {}
    for ann in existing_annotations:
        if not isinstance(ann, dict):
            continue
        task_name = ann.get("task_L2")
        if not task_name or task_name not in available_tasks:
            continue
        try:
            annotator = annotator_cache.get(task_name)
            if annotator is None:
                annotator = TaskAnnotatorFactory.create_annotator(
                    task_name=task_name,
                    gemini_client=gemini_client,
                    prompt_loader=prompt_loader,
                    bbox_annotator=bbox_annotator,
                    tracker=tracker,
                )
                annotator_cache[task_name] = annotator
            is_valid, error = annotator.validate_annotation(ann)
            if is_valid:
                completed_tasks.add(task_name)
            else:
                logger.warning(
                    "Existing annotation for %s is invalid: %s", task_name, error
                )
        except Exception as exc:
            logger.warning(
                "Failed to validate existing annotation for %s: %s", task_name, exc
            )

    tasks_to_run = [
        task_name
        for task_name in requested_tasks
        if task_name in available_tasks and task_name not in completed_tasks
    ]

    if not tasks_to_run:
        logger.info(
            "All known tasks already annotated for %s: %s",
            segment_type,
            segment_metadata.id,
        )
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
                tracker=tracker,
            )

            # Perform annotation
            annotation = annotator.annotate(segment_metadata, dataset_root=dataset_root)

            # Validate annotation
            is_valid, error = annotator.validate_annotation(annotation)
            if not is_valid:
                logger.warning(f"Invalid annotation for {task_name}: {error}")
                continue

            annotation = _maybe_write_tracking_mot(
                annotation, segment_metadata, output_dir
            )
            annotations.append(annotation)
            logger.info(f"Successfully annotated {task_name}")

        except NotImplementedError as e:
            logger.warning(f"Task {task_name} requires unimplemented features: {e}")
            continue
        except ValueError as e:
            logger.warning(f"Skipping task {task_name}: {e}")
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
            "annotations": annotations,
        }

    JSONUtils.save_json(output_data, output_path)
    logger.info(f"Saved {len(annotations)} annotations to {output_path}")

    return output_path


def _maybe_write_tracking_mot(
    annotation: Dict[str, Any], segment_metadata: ClipMetadata, output_dir: Path
) -> Dict[str, Any]:
    """Convert tracking_bboxes into MOT file reference when applicable."""
    tracking = annotation.get("tracking_bboxes")
    if not isinstance(tracking, dict):
        return annotation

    mot_rows = _tracking_to_mot_rows(tracking)
    if not mot_rows:
        return annotation

    config = get_config()
    if output_dir.name == "frames":
        return annotation
    if output_dir.name == "clips":
        mot_dir = output_dir / "mot"
    else:
        mot_dir = output_dir / "mot"
    mot_dir.mkdir(parents=True, exist_ok=True)

    task_name = annotation.get("task_L2", "tracking")
    mot_path = mot_dir / f"{segment_metadata.id}_{task_name}.txt"
    mot_path.write_text("\n".join(mot_rows) + "\n", encoding="utf-8")

    try:
        mot_ref = str(mot_path.relative_to(Path(config.project_root)))
    except ValueError:
        mot_ref = str(mot_path)

    annotation["tracking_bboxes"] = {"mot_file": mot_ref, "format": "MOTChallenge"}
    return annotation


def _tracking_to_mot_rows(tracking: Dict[str, Any]) -> List[str]:
    """Convert tracking result dict to MOTChallenge rows."""
    objects = tracking.get("objects", [])
    if not isinstance(objects, list) or not objects:
        return []

    frame_numbers: List[int] = []
    for obj in objects:
        frames = obj.get("frames", {})
        for frame_key in frames.keys():
            try:
                frame_numbers.append(int(frame_key))
            except (TypeError, ValueError):
                continue

    if not frame_numbers:
        return []

    min_frame = min(frame_numbers)
    frame_offset = 1 - min_frame if min_frame <= 0 else 0

    rows: List[str] = []
    for obj in objects:
        obj_id = int(obj.get("id", 0)) + 1
        frames = obj.get("frames", {})
        sortable_frames: List[Tuple[int, Any]] = []
        for frame_key in frames.keys():
            try:
                sortable_frames.append((int(frame_key), frame_key))
            except (TypeError, ValueError):
                continue
        for frame_int, frame_key in sorted(sortable_frames, key=lambda item: item[0]):
            frame_idx = frame_int + frame_offset
            bbox = frames.get(frame_key)
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            xtl, ytl, xbr, ybr = bbox
            width = xbr - xtl
            height = ybr - ytl
            if width <= 0 or height <= 0:
                continue
            row = (
                f"{frame_idx},{obj_id},"
                f"{xtl:.2f},{ytl:.2f},{width:.2f},{height:.2f},"
                "-1,-1,-1,-1"
            )
            rows.append(row)

    return rows


def _load_segment_metadata(
    segment_paths: Iterable[Path],
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


def _prune_orphan_outputs(output_dir: Path, valid_clip_ids: set[str]):
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
    segment_paths: List[Path], output_dir: Path, prune_orphans: bool = False
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
    bbox_annotator = BBoxAnnotator(gemini_client)
    model_path_value = config.tasks.tracking.get("model_path")
    model_path = None
    if model_path_value:
        model_path = Path(model_path_value)
        if not model_path.is_absolute():
            model_path = Path(config.project_root) / model_path

    tracker = ObjectTracker(
        backend=config.tasks.tracking.get("tracker_backend", "local"),
        model_path=model_path,
        hf_model_id=config.tasks.tracking.get("hf_model_id"),
        auto_download=config.tasks.tracking.get("auto_download", False),
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata upfront to support incremental behavior
    loaded_segments = _load_segment_metadata(segment_paths)
    valid_clip_ids = {metadata.id for _, metadata in loaded_segments}

    if prune_orphans and loaded_segments:
        _prune_orphan_outputs(output_dir, valid_clip_ids)

    if loaded_segments:
        clip_paths = [
            metadata.get_video_path(config.dataset_root)
            for _, metadata in loaded_segments
            if metadata.info.is_clip()
        ]
        if clip_paths:
            gemini_client.sync_gcs_objects(clip_paths)

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
                dataset_root=config.dataset_root,
            )

            successful += 1
            logger.info(f"Successfully processed clip: {segment_metadata.id}")

        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {segment_path}: {e}", exc_info=True)
            continue

    # Summary
    logger.info("=" * 60)
    logger.info("Processing complete")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {successful + failed}")
    logger.info("=" * 60)
    logger.info("AutoAnnotator finished")
