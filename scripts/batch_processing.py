from pathlib import Path
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from auto_annotator import InputAdapter
from auto_annotator.main import process_segment, setup_logging
from auto_annotator import GeminiClient, PromptLoader
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.config import get_config

setup_logging()
logger = logging.getLogger(__name__)
config = get_config()
dataset_root = Path(config.dataset_root)

model_path_value = config.tasks.tracking.get("model_path")
model_path = None
if model_path_value:
    model_path = Path(model_path_value)
    if not model_path.is_absolute():
        model_path = Path(config.project_root) / model_path


class LockedObjectTracker(ObjectTracker):
    def __init__(
        self,
        *args,
        lock: threading.Lock,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._lock = lock

    def track_from_first_bbox(self, *args, **kwargs):
        with self._lock:
            return super().track_from_first_bbox(*args, **kwargs)

    def track_with_query(self, *args, **kwargs):
        with self._lock:
            return super().track_with_query(*args, **kwargs)


tracking_lock = threading.Lock()
shared_tracker = LockedObjectTracker(
    backend=config.tasks.tracking.get("tracker_backend", "local"),
    model_path=model_path,
    hf_model_id=config.tasks.tracking.get("hf_model_id"),
    auto_download=config.tasks.tracking.get("auto_download", False),
    lock=tracking_lock,
)

thread_local = threading.local()


def get_thread_components():
    gemini_client = getattr(thread_local, "gemini_client", None)
    if gemini_client is None:
        gemini_client = GeminiClient()
        thread_local.gemini_client = gemini_client
        thread_local.prompt_loader = PromptLoader()
        thread_local.bbox_annotator = BBoxAnnotator(gemini_client)

    return (
        thread_local.gemini_client,
        thread_local.prompt_loader,
        thread_local.bbox_annotator,
        shared_tracker,
    )


def iter_metadata_files(event_dir: Path, enabled_kinds: tuple[str, ...]) -> list[Path]:
    metadata_files: list[Path] = []
    for sub_dir in enabled_kinds:
        candidate_dir = event_dir / sub_dir
        if not candidate_dir.exists():
            continue
        for json_path in candidate_dir.glob("*.json"):
            if json_path.stem.startswith("annotation_"):
                continue
            metadata_files.append(json_path)
    return metadata_files


def get_output_dir(metadata) -> Path:
    sub_dir = "frames" if metadata.info.is_single_frame() else "clips"
    return (
        Path(config.project_root)
        / config.output.temp_dir
        / metadata.origin.sport
        / metadata.origin.event
        / sub_dir
    )


batch_cfg = getattr(config, "batch_processing", None)
num_workers_value = 1
if batch_cfg is not None:
    num_workers_value = int(getattr(batch_cfg, "num_workers", 1) or 1)
if num_workers_value < 1:
    num_workers_value = 1

enable_clips = True
enable_frames = True
if batch_cfg is not None:
    enable_clips = bool(getattr(batch_cfg, "enable_clips", True))
    enable_frames = bool(getattr(batch_cfg, "enable_frames", True))

enabled_kinds: list[str] = []
if enable_clips:
    enabled_kinds.append("clips")
if enable_frames:
    enabled_kinds.append("frames")
if not enabled_kinds:
    logger.error(
        "Both batch_processing.enable_clips and batch_processing.enable_frames are false; nothing to do"
    )
    raise SystemExit(2)

enabled_kinds_tuple = tuple(enabled_kinds)


def process_one(json_path: Path, metadata) -> Path:
    gemini_client, prompt_loader, bbox_annotator, tracker = get_thread_components()
    return process_segment(
        segment_metadata=metadata,
        gemini_client=gemini_client,
        prompt_loader=prompt_loader,
        bbox_annotator=bbox_annotator,
        tracker=tracker,
        output_dir=get_output_dir(metadata),
        dataset_root=dataset_root,
    )


# 遍历所有运动项目
for sport_dir in dataset_root.iterdir():
    if not sport_dir.is_dir():
        continue

    logger.info("处理运动项目: %s", sport_dir.name)

    # 遍历所有比赛事件
    for event_dir in sport_dir.iterdir():
        if not event_dir.is_dir():
            continue

        logger.info("处理事件: %s", event_dir.name)

        metadata_files = iter_metadata_files(
            event_dir, enabled_kinds=enabled_kinds_tuple
        )
        metadata_entries = []
        for json_path in metadata_files:
            try:
                metadata = InputAdapter.load_from_json(json_path)
            except Exception as e:
                logger.warning("读取失败: %s: %s", json_path, e)
                continue

            if str(metadata.id) != json_path.stem:
                logger.warning("ID mismatch: %s has id=%s", json_path, metadata.id)

            ok, err = InputAdapter.validate_metadata(
                metadata, dataset_root=dataset_root
            )
            if not ok:
                logger.error(
                    "Invalid clip metadata: %s (id=%s) -> %s",
                    json_path,
                    metadata.id,
                    err,
                )
                continue

            metadata_entries.append((json_path, metadata))

        logger.info("找到 %s 个片段/单帧", len(metadata_entries))

        if num_workers_value == 1:
            for json_path, metadata in metadata_entries:
                try:
                    output_path = process_one(json_path, metadata)
                    logger.info("✓ %s (%s) -> %s", metadata.id, json_path, output_path)
                except Exception as e:
                    logger.error("✗ %s (%s): %s", metadata.id, json_path, e)
        else:
            with ThreadPoolExecutor(max_workers=num_workers_value) as executor:
                future_map = {
                    executor.submit(process_one, json_path, metadata): (
                        json_path,
                        metadata,
                    )
                    for json_path, metadata in metadata_entries
                }

                for fut in as_completed(future_map):
                    json_path, metadata = future_map[fut]
                    try:
                        output_path = fut.result()
                        logger.info(
                            "✓ %s (%s) -> %s", metadata.id, json_path, output_path
                        )
                    except Exception as e:
                        logger.error("✗ %s (%s): %s", metadata.id, json_path, e)

logger.info("批量处理完成！")
