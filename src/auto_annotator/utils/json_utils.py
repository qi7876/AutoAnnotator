"""JSON processing utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class JSONUtils:
    """Utilities for JSON file operations."""

    @staticmethod
    def load_json(json_path: Path) -> Dict[str, Any]:
        """
        Load JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Loaded JSON from {json_path}")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}")

    @staticmethod
    def save_json(
        data: Dict[str, Any],
        json_path: Path,
        indent: int = 2,
        ensure_ascii: bool = False
    ):
        """
        Save data to JSON file.

        Args:
            data: Data to save
            json_path: Output path
            indent: JSON indentation (default: 2)
            ensure_ascii: Whether to escape non-ASCII characters (default: False)
        """
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.debug(f"Saved JSON to {json_path}")

    @staticmethod
    def merge_annotations(
        base_json: Dict[str, Any],
        new_annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge new annotations into base JSON.

        Args:
            base_json: Base annotation JSON
            new_annotations: New annotations to add

        Returns:
            Merged JSON
        """
        result = base_json.copy()

        # Ensure annotations field exists
        if "annotations" not in result:
            result["annotations"] = []

        # Get next annotation ID
        existing_ids = [
            int(ann.get("annotation_id", 0))
            for ann in result["annotations"]
            if ann.get("annotation_id", "").isdigit()
        ]
        next_id = max(existing_ids) + 1 if existing_ids else 1

        # Add new annotations with updated IDs
        for ann in new_annotations:
            ann_copy = ann.copy()
            ann_copy["annotation_id"] = str(next_id)
            result["annotations"].append(ann_copy)
            next_id += 1

        logger.info(f"Merged {len(new_annotations)} annotations")
        return result

    @staticmethod
    def validate_annotation_json(data: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate annotation JSON structure.

        Args:
            data: JSON data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["sport", "event", "video_id", "video_metadata", "annotations"]

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Validate video_metadata
        metadata = data["video_metadata"]
        required_metadata = ["duration_sec", "fps", "total_frames", "resolution"]

        for field in required_metadata:
            if field not in metadata:
                return False, f"Missing video_metadata field: {field}"

        # Validate annotations is a list
        if not isinstance(data["annotations"], list):
            return False, "annotations must be a list"

        return True, ""

    @staticmethod
    def get_annotation_ids(json_data: Dict[str, Any]) -> List[str]:
        """
        Get all annotation IDs from a JSON file.

        Args:
            json_data: Annotation JSON data

        Returns:
            List of annotation IDs
        """
        annotations = json_data.get("annotations", [])
        return [ann.get("annotation_id", "") for ann in annotations]

    @staticmethod
    def filter_annotations_by_task(
        json_data: Dict[str, Any],
        task_l1: str = None,
        task_l2: str = None
    ) -> List[Dict[str, Any]]:
        """
        Filter annotations by task type.

        Args:
            json_data: Annotation JSON data
            task_l1: Level 1 task name (e.g., "Understanding", "Perception")
            task_l2: Level 2 task name (e.g., "ScoreboardSingle")

        Returns:
            List of filtered annotations
        """
        annotations = json_data.get("annotations", [])
        filtered = []

        for ann in annotations:
            if task_l1 and ann.get("task_L1") != task_l1:
                continue
            if task_l2 and ann.get("task_L2") != task_l2:
                continue
            filtered.append(ann)

        return filtered

    @staticmethod
    def create_base_annotation_json(
        sport: str,
        event: str,
        video_id: str,
        video_metadata: Dict[str, Any],
        info: str = ""
    ) -> Dict[str, Any]:
        """
        Create a base annotation JSON structure.

        Args:
            sport: Sport name
            event: Event name
            video_id: Video ID
            video_metadata: Video metadata dictionary
            info: Optional video description

        Returns:
            Base annotation JSON structure
        """
        return {
            "sport": sport,
            "event": event,
            "video_id": video_id,
            "info": info,
            "video_metadata": video_metadata,
            "annotations": []
        }
