#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Optional

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from auto_annotator import ClipMetadata, GeminiClient, InputAdapter, JSONUtils

logger = logging.getLogger(__name__)

AI_COACH_TASK = "AI_Coach"
DEFAULT_DATASET_ROOT = Path("data/Dataset")
DEFAULT_OUTPUT_ROOT = Path("data/output")
DEFAULT_PROMPT_PATH = Path("config/prompts/ai_coach.md")
DEFAULT_LANGUAGE = "en"
DEFAULT_LANGUAGE_INSTRUCTION = "Use English for both questions and answers."

_QA_TEXT_PATTERN = re.compile(
    r"(?:Q|Question|问)\s*[:：]\s*(?P<question>.+?)\s*"
    r"(?:A|Answer|答)\s*[:：]\s*(?P<answer>.+)",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class BatchStats:
    scanned_jsons: int = 0
    matched_ai_coach: int = 0
    skipped_existing: int = 0
    annotated: int = 0
    failed: int = 0


@dataclass(frozen=True)
class ClipJob:
    json_path: Path
    metadata: ClipMetadata
    output_path: Path
    existing_output: Optional[dict[str, Any]]


def _read_json_dict(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata must be a JSON object: {json_path}")
    return data


def _load_clip_metadata(json_path: Path) -> ClipMetadata:
    data = _read_json_dict(json_path)
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


def build_ai_coach_prompt(
    prompt_template: str,
    metadata: ClipMetadata,
) -> str:
    total_frames = metadata.info.total_frames
    fps = metadata.info.fps
    duration_sec = total_frames / fps
    return prompt_template.format(
        total_frames=total_frames,
        max_frame=max(0, total_frames - 1),
        fps=fps,
        duration_sec=duration_sec,
        num_first_frame=metadata.info.original_starting_frame,
        language=DEFAULT_LANGUAGE,
        language_instruction=DEFAULT_LANGUAGE_INSTRUCTION,
    )


def _extract_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _parse_qa_text(value: str) -> dict[str, str]:
    text = value.strip()
    match = _QA_TEXT_PATTERN.match(text)
    if match is not None:
        question = match.group("question").strip()
        answer = match.group("answer").strip()
        if question and answer:
            return {"question": question, "answer": answer}
    raise ValueError(f"Invalid QA text item: {value}")


def _normalize_qa_item(item: Any) -> dict[str, str]:
    if isinstance(item, dict):
        question = (
            _extract_text(item.get("question"))
            or _extract_text(item.get("q"))
            or _extract_text(item.get("Q"))
        )
        answer = (
            _extract_text(item.get("answer"))
            or _extract_text(item.get("a"))
            or _extract_text(item.get("A"))
        )
        if question and answer:
            return {"question": question, "answer": answer}
        raise ValueError(f"Invalid QA dict item: {item}")

    if isinstance(item, str):
        return _parse_qa_text(item)

    raise ValueError(f"Unsupported QA item type: {type(item).__name__}")


def _normalize_qa_container(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return [_normalize_qa_item(item) for item in value]
    if isinstance(value, dict):
        return [_normalize_qa_item(value)]
    if isinstance(value, str):
        return [_normalize_qa_item(value)]
    raise ValueError(f"Unsupported QA container type: {type(value).__name__}")


def normalize_ai_coach_response(result: Any) -> list[dict[str, str]]:
    if isinstance(result, dict):
        for key in ("qa_pairs", "qas", "qa", "pairs", "items"):
            if key in result:
                qa_pairs = _normalize_qa_container(result[key])
                if qa_pairs:
                    return qa_pairs
                raise ValueError("qa_pairs is empty")

        if (
            ("question" in result or "q" in result or "Q" in result)
            and ("answer" in result or "a" in result or "A" in result)
        ):
            return [_normalize_qa_item(result)]
        raise ValueError("Response object does not contain QA pairs")

    if isinstance(result, list):
        qa_pairs = _normalize_qa_container(result)
        if qa_pairs:
            return qa_pairs
        raise ValueError("Response list is empty")

    raise ValueError(f"Unsupported response type: {type(result).__name__}")


def keep_only_one_qa_pair(qa_pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    if not qa_pairs:
        raise ValueError("qa_pairs is empty")
    if len(qa_pairs) > 1:
        logger.warning(
            "Model returned %s QA pairs; keeping only the first one.",
            len(qa_pairs),
        )
    return [qa_pairs[0]]


def _has_completed_ai_coach(existing_output: dict[str, Any]) -> bool:
    annotations = existing_output.get("annotations")
    if not isinstance(annotations, list):
        return False
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("task_L2") != AI_COACH_TASK:
            continue
        qa_pairs = annotation.get("qa_pairs")
        if not isinstance(qa_pairs, list) or not qa_pairs:
            continue
        try:
            _normalize_qa_container(qa_pairs)
            return True
        except ValueError:
            continue
    return False


def _build_ai_coach_annotation(qa_pairs: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "task_L1": "Understanding",
        "task_L2": AI_COACH_TASK,
        "reviewed": False,
        "qa_pairs": qa_pairs,
    }


def _build_output_data(
    metadata: ClipMetadata,
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
                if item.get("task_L2") == AI_COACH_TASK:
                    continue
                merged_annotations.append(dict(item))
    merged_annotations.append(dict(annotation))

    for index, item in enumerate(merged_annotations, start=1):
        item["annotation_id"] = str(index)

    output: dict[str, Any] = (
        dict(existing_output) if isinstance(existing_output, dict) else {}
    )
    output["id"] = metadata.id
    output["origin"] = {
        "sport": metadata.origin.sport,
        "event": metadata.origin.event,
    }
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
        "matched_ai_coach": stats.matched_ai_coach,
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
            metadata = _load_clip_metadata(json_path)
        except Exception as exc:
            logger.warning("Skip invalid metadata %s: %s", json_path, exc)
            stats = _inc_stats(stats, failed=stats.failed + 1)
            continue

        if metadata.info.is_single_frame():
            continue

        if AI_COACH_TASK not in metadata.tasks_to_annotate:
            continue

        stats = _inc_stats(stats, matched_ai_coach=stats.matched_ai_coach + 1)

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
                and _has_completed_ai_coach(existing_output)
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
        prompt = build_ai_coach_prompt(
            prompt_template=prompt_template,
            metadata=job.metadata,
        )
        video_path = job.metadata.get_video_path(dataset_root)
        video_file = gemini_client.upload_video(video_path)
        raw_result = gemini_client.annotate_video(video_file, prompt)
        qa_pairs = keep_only_one_qa_pair(normalize_ai_coach_response(raw_result))
        annotation = _build_ai_coach_annotation(qa_pairs)
        output_data = _build_output_data(
            metadata=job.metadata,
            annotation=annotation,
            existing_output=job.existing_output,
        )
        JSONUtils.save_json(output_data, job.output_path)
        logger.info(
            "Annotated AI_Coach: %s/%s/%s -> %s QA pairs",
            job.metadata.origin.sport,
            job.metadata.origin.event,
            job.metadata.id,
            len(qa_pairs),
        )
        return True, job.json_path
    except Exception as exc:
        logger.error("Failed to annotate %s: %s", job.json_path, exc)
        return False, job.json_path
    finally:
        if video_file is not None:
            gemini_client.cleanup_file(video_file)


def annotate_ai_coach_batch(
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

    if workers == 1:
        client = gemini_client if gemini_client is not None else gemini_client_factory()
        iterator: Iterable[ClipJob] = jobs
        progress_bar = None
        if use_tqdm:
            progress_bar = tqdm(total=len(jobs), desc="AI_Coach", unit="clip")
        try:
            for job in iterator:
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

    progress_bar = tqdm(total=len(jobs), desc="AI_Coach", unit="clip") if use_tqdm else None
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_run_job_with_thread_client, job): job
                for job in jobs
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


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AI_Coach QA annotations for clip metadata."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory (default: data/Dataset).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (default: data/output).",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Prompt template path (default: config/prompts/ai_coach.md).",
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
        help="Maximum number of AI_Coach clips to annotate in this run.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers for annotation (default: min(8, CPU cores)).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate AI_Coach even if an existing AI_Coach annotation exists.",
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
    args = parse_args(argv)
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

    stats = annotate_ai_coach_batch(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        gemini_client_factory=GeminiClient,
        prompt_template=prompt_template,
        overwrite=bool(args.overwrite),
        sport=args.sport,
        event=args.event,
        limit=args.limit,
        num_workers=args.num_workers,
        progress=not args.no_progress,
    )

    logger.info(
        "Finished AI_Coach batch. scanned=%s matched=%s annotated=%s skipped_existing=%s failed=%s",
        stats.scanned_jsons,
        stats.matched_ai_coach,
        stats.annotated,
        stats.skipped_existing,
        stats.failed,
    )
    return 1 if stats.failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
