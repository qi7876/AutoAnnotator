#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from auto_annotator import (
    ClipMetadata,
    GeminiClient,
    InputAdapter,
    JSONUtils,
    get_config,
)

logger = logging.getLogger(__name__)

SPATIAL_IMAGINATION_TASK = "Spatial_Imagination"
DEFAULT_PROMPT_PATH = Path("config/prompts/spatial_imagination.md")

_QA_TEXT_PATTERN = re.compile(
    r"(?:Q|Question|问)\s*[:：]\s*(?P<question>.+?)\s*"
    r"(?:A|Answer|答)\s*[:：]\s*(?P<answer>.+)",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class BatchStats:
    scanned_jsons: int = 0
    matched_spatial_imagination: int = 0
    skipped_existing: int = 0
    annotated: int = 0
    failed: int = 0


@dataclass(frozen=True)
class ClipJob:
    json_path: Path
    metadata: ClipMetadata
    raw_data: dict[str, Any]
    output_path: Path
    existing_output: Optional[dict[str, Any]]


def _read_json_dict(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata must be a JSON object: {json_path}")
    return data


def _load_clip_metadata(raw_data: dict[str, Any]) -> ClipMetadata:
    data = raw_data
    if "tasks_to_annotate" not in data and "task_to_annotate" in data:
        data = dict(data)
        data["tasks_to_annotate"] = data["task_to_annotate"]
    return InputAdapter.create_from_dict(data)


def _iter_clip_metadata_jsons(
    dataset_root: Path,
    sport: Optional[str] = None,
    event: Optional[str] = None,
) -> Iterable[Path]:
    for sport_dir in sorted(dataset_root.iterdir()):
        if not sport_dir.is_dir():
            continue
        if sport is not None and sport_dir.name != sport:
            continue
        for event_dir in sorted(sport_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            if event is not None and event_dir.name != event:
                continue
            clips_dir = event_dir / "clips"
            if not clips_dir.exists():
                continue
            yield from sorted(clips_dir.glob("*.json"))


def load_prompt_template(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _extract_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _extract_source_context(raw_data: dict[str, Any]) -> dict[str, Any]:
    source = raw_data.get("source_annotation")
    if not isinstance(source, dict):
        raise ValueError("source_annotation is missing or invalid.")

    source_task_l2 = _extract_text(source.get("task_L2"))
    source_annotation_id = _extract_text(source.get("source_annotation_id"))

    ann = source.get("annotation")
    if not isinstance(ann, dict):
        ann = {}

    source_question = _extract_text(ann.get("question"))
    source_query = _extract_text(ann.get("query"))
    source_first_frame_description = _extract_text(ann.get("first_frame_description"))

    source_object_reference = (
        source_query or source_first_frame_description or source_question
    )
    if not source_object_reference:
        raise ValueError(
            "Cannot extract source object reference from source_annotation.annotation."
        )

    source_answer = ann.get("answer")
    if isinstance(source_answer, list):
        source_answer = [item for item in source_answer if isinstance(item, str)]
    elif not isinstance(source_answer, str):
        source_answer = ""

    return {
        "source_annotation_id": source_annotation_id,
        "source_task_L2": source_task_l2,
        "source_object_reference": source_object_reference,
        "source_question": source_question,
        "source_query": source_query,
        "source_first_frame_description": source_first_frame_description,
        "source_answer": source_answer,
    }


def build_spatial_imagination_prompt(
    prompt_template: str,
    metadata: ClipMetadata,
    raw_data: dict[str, Any],
) -> str:
    source_context = _extract_source_context(raw_data)
    total_frames = metadata.info.total_frames
    fps = metadata.info.fps
    duration_sec = total_frames / fps
    return prompt_template.format(
        total_frames=total_frames,
        max_frame=max(0, total_frames - 1),
        fps=fps,
        duration_sec=duration_sec,
        num_first_frame=metadata.info.original_starting_frame,
        source_task_l2=source_context["source_task_L2"],
        source_object_reference=source_context["source_object_reference"],
        source_context_json=json.dumps(source_context, ensure_ascii=False, indent=2),
    )


def _parse_qa_text(value: str) -> tuple[str, str]:
    text = value.strip()
    match = _QA_TEXT_PATTERN.match(text)
    if match is not None:
        question = match.group("question").strip()
        answer = match.group("answer").strip()
        if question and answer:
            return question, answer
    raise ValueError(f"Invalid QA text item: {value}")


def _normalize_question_answer(question: Any, answer: Any) -> tuple[str, str]:
    question_text = _extract_text(question)
    if isinstance(answer, list):
        answer_parts = [item.strip() for item in answer if isinstance(item, str) and item.strip()]
        answer_text = " ".join(answer_parts).strip()
    else:
        answer_text = _extract_text(answer)
    if not question_text or not answer_text:
        raise ValueError("Question or answer is empty.")
    return question_text, answer_text


def normalize_spatial_imagination_response(result: Any) -> tuple[str, str]:
    if isinstance(result, dict):
        if "question" in result and "answer" in result:
            return _normalize_question_answer(result.get("question"), result.get("answer"))

        qa = result.get("qa")
        if isinstance(qa, dict):
            return _normalize_question_answer(qa.get("question"), qa.get("answer"))

        for key in ("qa_pairs", "qas", "items"):
            value = result.get(key)
            if isinstance(value, list) and value:
                return normalize_spatial_imagination_response(value[0])
            if isinstance(value, dict):
                return normalize_spatial_imagination_response(value)
            if isinstance(value, str):
                return _parse_qa_text(value)

        raise ValueError("Response object does not contain question/answer.")

    if isinstance(result, list):
        if not result:
            raise ValueError("Response list is empty.")
        return normalize_spatial_imagination_response(result[0])

    if isinstance(result, str):
        return _parse_qa_text(result)

    raise ValueError(f"Unsupported response type: {type(result).__name__}")


def _has_completed_spatial_imagination(existing_output: dict[str, Any]) -> bool:
    annotations = existing_output.get("annotations")
    if not isinstance(annotations, list):
        return False
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("task_L2") != SPATIAL_IMAGINATION_TASK:
            continue
        question = _extract_text(annotation.get("question"))
        answer = _extract_text(annotation.get("answer"))
        if question and answer:
            return True
    return False


def _build_annotation(question: str, answer: str) -> dict[str, Any]:
    return {
        "task_L1": "Understanding",
        "task_L2": SPATIAL_IMAGINATION_TASK,
        "reviewed": False,
        "question": question,
        "answer": answer,
    }


def _build_output_data(
    *,
    metadata: ClipMetadata,
    raw_data: dict[str, Any],
    annotation: dict[str, Any],
    existing_output: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    merged_annotations: list[dict[str, Any]] = []
    if existing_output is not None:
        existing_annotations = existing_output.get("annotations")
        if isinstance(existing_annotations, list):
            for item in existing_annotations:
                if not isinstance(item, dict):
                    continue
                if item.get("task_L2") == SPATIAL_IMAGINATION_TASK:
                    continue
                merged_annotations.append(dict(item))
    merged_annotations.append(dict(annotation))

    for index, item in enumerate(merged_annotations, start=1):
        item["annotation_id"] = str(index)

    output: dict[str, Any] = (
        dict(existing_output) if isinstance(existing_output, dict) else {}
    )
    output["id"] = metadata.id

    raw_origin = raw_data.get("origin")
    origin = {
        "sport": metadata.origin.sport,
        "event": metadata.origin.event,
    }
    if isinstance(raw_origin, dict):
        video_id = raw_origin.get("video_id")
        if isinstance(video_id, (str, int)):
            origin["video_id"] = str(video_id)
    output["origin"] = origin
    output["annotations"] = merged_annotations
    return output


def get_output_path(output_root: Path, metadata: ClipMetadata) -> Path:
    return (
        output_root
        / metadata.origin.sport
        / metadata.origin.event
        / "clips"
        / f"{metadata.id}.json"
    )


def _inc_stats(stats: BatchStats, **updates: int) -> BatchStats:
    values = {
        "scanned_jsons": stats.scanned_jsons,
        "matched_spatial_imagination": stats.matched_spatial_imagination,
        "skipped_existing": stats.skipped_existing,
        "annotated": stats.annotated,
        "failed": stats.failed,
    }
    for key, value in updates.items():
        values[key] = value
    return BatchStats(**values)


def _prepare_jobs(
    *,
    dataset_root: Path,
    output_root: Path,
    overwrite: bool,
    sport: Optional[str],
    event: Optional[str],
    limit: Optional[int],
) -> tuple[list[ClipJob], BatchStats]:
    jobs: list[ClipJob] = []
    stats = BatchStats()

    for json_path in _iter_clip_metadata_jsons(dataset_root, sport=sport, event=event):
        stats = _inc_stats(stats, scanned_jsons=stats.scanned_jsons + 1)

        try:
            raw_data = _read_json_dict(json_path)
            metadata = _load_clip_metadata(raw_data)
            _extract_source_context(raw_data)
        except Exception as exc:
            logger.warning("Skip invalid Spatial_Imagination metadata %s: %s", json_path, exc)
            stats = _inc_stats(stats, failed=stats.failed + 1)
            continue

        if metadata.info.is_single_frame():
            continue

        if SPATIAL_IMAGINATION_TASK not in metadata.tasks_to_annotate:
            continue
        stats = _inc_stats(
            stats,
            matched_spatial_imagination=stats.matched_spatial_imagination + 1,
        )

        output_path = get_output_path(output_root, metadata)
        existing_output: Optional[dict[str, Any]] = None
        if output_path.exists():
            try:
                existing_output = JSONUtils.load_json(output_path)
            except Exception as exc:
                logger.warning("Failed to parse existing output %s: %s", output_path, exc)
            if (
                not overwrite
                and existing_output is not None
                and _has_completed_spatial_imagination(existing_output)
            ):
                stats = _inc_stats(stats, skipped_existing=stats.skipped_existing + 1)
                continue

        valid, error = InputAdapter.validate_metadata(metadata, dataset_root=dataset_root)
        if not valid:
            logger.warning("Skip invalid clip metadata %s: %s", json_path, error)
            stats = _inc_stats(stats, failed=stats.failed + 1)
            continue

        jobs.append(
            ClipJob(
                json_path=json_path,
                metadata=metadata,
                raw_data=raw_data,
                output_path=output_path,
                existing_output=existing_output,
            )
        )
        if limit is not None and len(jobs) >= limit:
            break
    return jobs, stats


def _run_one_job(
    *,
    job: ClipJob,
    dataset_root: Path,
    prompt_template: str,
    gemini_client: Any,
) -> tuple[bool, Path]:
    video_file: Any = None
    try:
        prompt = build_spatial_imagination_prompt(
            prompt_template=prompt_template,
            metadata=job.metadata,
            raw_data=job.raw_data,
        )
        video_path = job.metadata.get_video_path(dataset_root)
        video_file = gemini_client.upload_video(video_path)
        raw_result = gemini_client.annotate_video(video_file, prompt)
        question, answer = normalize_spatial_imagination_response(raw_result)
        annotation = _build_annotation(question, answer)
        output_data = _build_output_data(
            metadata=job.metadata,
            raw_data=job.raw_data,
            annotation=annotation,
            existing_output=job.existing_output,
        )
        JSONUtils.save_json(output_data, job.output_path)
        logger.info(
            "Annotated Spatial_Imagination: %s/%s/%s",
            job.metadata.origin.sport,
            job.metadata.origin.event,
            job.metadata.id,
        )
        return True, job.json_path
    except Exception as exc:
        logger.error("Failed to annotate %s: %s", job.json_path, exc)
        return False, job.json_path
    finally:
        if video_file is not None:
            gemini_client.cleanup_file(video_file)


def annotate_spatial_imagination_batch(
    *,
    dataset_root: Path,
    output_root: Path,
    gemini_client: Optional[Any] = None,
    gemini_client_factory: Optional[Callable[[], Any]] = None,
    prompt_template: str,
    overwrite: bool = False,
    sport: Optional[str] = None,
    event: Optional[str] = None,
    limit: Optional[int] = None,
    num_workers: int = 1,
    progress: bool = True,
) -> BatchStats:
    jobs, stats = _prepare_jobs(
        dataset_root=dataset_root,
        output_root=output_root,
        overwrite=overwrite,
        sport=sport,
        event=event,
        limit=limit,
    )
    if not jobs:
        return stats

    workers = max(1, int(num_workers))
    if gemini_client_factory is None:
        if gemini_client is None:
            raise ValueError(
                "Either gemini_client or gemini_client_factory must be provided."
            )
        if workers > 1:
            logger.warning(
                "num_workers=%s requested, but no gemini_client_factory provided; "
                "falling back to single-thread execution.",
                workers,
            )
            workers = 1

    use_tqdm = tqdm is not None and progress
    progress_bar = tqdm(total=len(jobs), desc="Spatial_Imagination", unit="clip") if use_tqdm else None

    if workers == 1:
        client = gemini_client if gemini_client is not None else gemini_client_factory()
        try:
            for job in jobs:
                ok, _ = _run_one_job(
                    job=job,
                    dataset_root=dataset_root,
                    prompt_template=prompt_template,
                    gemini_client=client,
                )
                if ok:
                    stats = _inc_stats(stats, annotated=stats.annotated + 1)
                else:
                    stats = _inc_stats(stats, failed=stats.failed + 1)
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return stats

    thread_local = threading.local()

    def _get_client() -> Any:
        client = getattr(thread_local, "gemini_client", None)
        if client is None:
            assert gemini_client_factory is not None
            client = gemini_client_factory()
            thread_local.gemini_client = client
        return client

    def _run_job_with_thread_client(job: ClipJob) -> tuple[bool, Path]:
        return _run_one_job(
            job=job,
            dataset_root=dataset_root,
            prompt_template=prompt_template,
            gemini_client=_get_client(),
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_run_job_with_thread_client, job): job for job in jobs
            }
            for future in as_completed(future_map):
                try:
                    ok, _ = future.result()
                except Exception as exc:
                    job = future_map[future]
                    logger.error("Unhandled worker failure on %s: %s", job.json_path, exc)
                    ok = False
                if ok:
                    stats = _inc_stats(stats, annotated=stats.annotated + 1)
                else:
                    stats = _inc_stats(stats, failed=stats.failed + 1)
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return stats


def parse_args(
    argv: Optional[list[str]],
    *,
    default_dataset_root: Path,
    default_output_root: Path,
    default_num_workers: int,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Spatial_Imagination annotations from clips based on source_annotation "
            "(uses AutoAnnotator config for Gemini backend/model/GCS and default concurrency)."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
        help=f"Dataset root directory (default: {default_dataset_root}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help=f"Output root directory (default: {default_output_root}).",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Prompt template path (default: config/prompts/spatial_imagination.md).",
    )
    parser.add_argument(
        "--sport",
        type=str,
        default=None,
        help="Optional sport directory filter.",
    )
    parser.add_argument(
        "--event",
        type=str,
        default=None,
        help="Optional event directory filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of Spatial_Imagination clips to annotate in this run.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, default_num_workers),
        help="Number of parallel workers (default from config.batch_processing.num_workers).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate Spatial_Imagination even if an existing annotation exists.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    config = get_config()
    default_dataset_root = Path(config.dataset_root)
    default_output_root = Path(config.project_root) / "data" / "output"
    default_num_workers = int(
        getattr(getattr(config, "batch_processing", None), "num_workers", 1) or 1
    )

    args = parse_args(
        argv,
        default_dataset_root=default_dataset_root,
        default_output_root=default_output_root,
        default_num_workers=default_num_workers,
    )
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.dataset_root.exists() or not args.dataset_root.is_dir():
        logger.error("dataset_root not found or not a directory: %s", args.dataset_root)
        return 2

    try:
        prompt_template = load_prompt_template(args.prompt_path)
    except Exception as exc:
        logger.error("Failed to load prompt template: %s", exc)
        return 2

    stats = annotate_spatial_imagination_batch(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        gemini_client_factory=GeminiClient,
        prompt_template=prompt_template,
        overwrite=bool(args.overwrite),
        sport=args.sport,
        event=args.event,
        limit=args.limit,
        num_workers=max(1, int(args.num_workers)),
        progress=not args.no_progress,
    )
    logger.info(
        "Finished Spatial_Imagination batch. scanned=%s matched=%s annotated=%s skipped_existing=%s failed=%s",
        stats.scanned_jsons,
        stats.matched_spatial_imagination,
        stats.annotated,
        stats.skipped_existing,
        stats.failed,
    )
    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
