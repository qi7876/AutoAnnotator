"""Base annotator class for all annotation tasks."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..adapters import ClipMetadata
from ..utils import PromptLoader
from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class BaseAnnotator(ABC):
    """Base class for all task annotators."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_loader: PromptLoader
    ):
        """
        Initialize base annotator.

        Args:
            gemini_client: Gemini API client
            prompt_loader: Prompt template loader
        """
        self.gemini_client = gemini_client
        self.prompt_loader = prompt_loader
        self.task_name = self.get_task_name()

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def get_task_name(self) -> str:
        """
        Get the task name for this annotator.

        Returns:
            Task name (e.g., "ScoreboardSingle")
        """
        raise NotImplementedError

    @abstractmethod
    def get_task_l1(self) -> str:
        """
        Get level 1 task category.

        Returns:
            Task L1 (e.g., "Understanding" or "Perception")
        """
        raise NotImplementedError

    @abstractmethod
    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Perform annotation for this task.

        Args:
            segment_metadata: Segment metadata
            dataset_root: Root directory of the dataset

        Returns:
            Annotation result dictionary
        """
        raise NotImplementedError

    def prepare_prompt_variables(
        self,
        segment_metadata: ClipMetadata
    ) -> Dict[str, Any]:
        """
        Prepare common variables for prompt template.

        Args:
            segment_metadata: Segment metadata

        Returns:
            Dictionary of template variables
        """
        return {
            "num_first_frame": segment_metadata.info.original_starting_frame,
            "total_frames": segment_metadata.info.total_frames,
            "fps": segment_metadata.info.fps,
            "duration_sec": segment_metadata.info.total_frames / segment_metadata.info.fps,
        }

    def load_prompt(
        self,
        segment_metadata: ClipMetadata,
        **extra_vars
    ) -> str:
        """
        Load and format prompt for this task.

        Args:
            segment_metadata: Segment metadata
            **extra_vars: Additional template variables

        Returns:
            Formatted prompt string
        """
        variables = self.prepare_prompt_variables(segment_metadata)
        variables.update(extra_vars)

        return self.prompt_loader.load_prompt(self.task_name, **variables)

    def add_metadata_fields(
        self,
        annotation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add standard metadata fields to annotation.

        Args:
            annotation: Raw annotation from AI

        Returns:
            Annotation with metadata fields added
        """
        # annotation_id will be set during merging
        annotation["task_L1"] = self.get_task_l1()
        annotation["task_L2"] = self.task_name
        # review flag for downstream QA tools
        annotation.setdefault("reviewed", False)

        return annotation

    def validate_annotation(
        self,
        annotation: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate annotation result.

        Args:
            annotation: Annotation to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation - can be overridden by subclasses
        if not isinstance(annotation, dict):
            return False, "Annotation must be a dictionary"

        return True, None

    def normalize_result(self, result: Any) -> Dict[str, Any]:
        """
        Normalize model output into a dict.

        Accepts a single dict or a list with one dict.
        """
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            return result[0]
        raise ValueError(f"Unexpected response format: {result}")
