"""Input adapter for handling segment metadata from previous steps."""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class OriginalVideoInfo(BaseModel):
    """Information about the original video."""

    path: Path
    json_path: Path
    sport: str
    event: str
    video_id: str

    @validator("path", "json_path", pre=True)
    def convert_to_path(cls, v):
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v


class SegmentInfo(BaseModel):
    """Information about the video segment."""

    path: Path
    start_frame_in_original: int
    total_frames: int
    fps: int = 10
    duration_sec: float
    resolution: List[int] = Field(default_factory=lambda: [1920, 1080])

    @validator("path", pre=True)
    def convert_to_path(cls, v):
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v)
        return v


class SegmentMetadata(BaseModel):
    """Complete metadata for a video segment."""

    segment_id: str
    original_video: OriginalVideoInfo
    segment_info: SegmentInfo
    tasks_to_annotate: List[str] = Field(default_factory=list)
    additional_info: dict = Field(default_factory=dict)

    def has_task(self, task_name: str) -> bool:
        """Check if a task should be annotated for this segment."""
        return task_name in self.tasks_to_annotate

    def get_video_path(self) -> Path:
        """Get the path to the segment video file."""
        return self.segment_info.path

    def get_original_video_path(self) -> Path:
        """Get the path to the original video file."""
        return self.original_video.path


class InputAdapter:
    """
    Adapter for handling different input formats from previous steps.

    This class provides a unified interface for loading segment metadata,
    regardless of the format used by the segment splitting step (step 1-2).
    """

    @staticmethod
    def load_from_json(json_path: Path) -> SegmentMetadata:
        """
        Load segment metadata from a JSON file.

        Args:
            json_path: Path to the segment metadata JSON file

        Returns:
            SegmentMetadata object

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON format is invalid
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Segment metadata file not found: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SegmentMetadata(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading segment metadata: {e}")

    @staticmethod
    def load_from_directory(segments_dir: Path) -> List[SegmentMetadata]:
        """
        Load all segment metadata files from a directory.

        Args:
            segments_dir: Directory containing segment metadata JSON files

        Returns:
            List of SegmentMetadata objects
        """
        if not segments_dir.exists():
            raise FileNotFoundError(f"Segments directory not found: {segments_dir}")

        metadata_list = []
        for json_file in segments_dir.glob("*.json"):
            # Skip if this is not a segment metadata file
            # (e.g., might be an annotation result file)
            if json_file.stem.startswith("annotation_"):
                continue

            try:
                metadata = InputAdapter.load_from_json(json_file)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue

        return metadata_list

    @staticmethod
    def create_from_dict(data: dict) -> SegmentMetadata:
        """
        Create SegmentMetadata from a dictionary.

        This method is useful for testing or when receiving data
        from other Python components.

        Args:
            data: Dictionary containing segment metadata

        Returns:
            SegmentMetadata object
        """
        return SegmentMetadata(**data)

    @staticmethod
    def validate_metadata(metadata: SegmentMetadata) -> tuple[bool, Optional[str]]:
        """
        Validate that segment metadata is complete and consistent.

        Args:
            metadata: SegmentMetadata to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check video file exists
        if not metadata.get_video_path().exists():
            return False, f"Segment video file not found: {metadata.get_video_path()}"

        # Check original video file exists
        if not metadata.get_original_video_path().exists():
            return False, f"Original video file not found: {metadata.get_original_video_path()}"

        # Check frame numbers are valid
        if metadata.segment_info.start_frame_in_original < 0:
            return False, "start_frame_in_original must be non-negative"

        if metadata.segment_info.total_frames <= 0:
            return False, "total_frames must be positive"

        # Check FPS is valid
        if metadata.segment_info.fps <= 0:
            return False, "fps must be positive"

        # Check duration is consistent with frames and fps
        expected_duration = metadata.segment_info.total_frames / metadata.segment_info.fps
        if abs(expected_duration - metadata.segment_info.duration_sec) > 0.1:
            return False, (
                f"Inconsistent duration: expected {expected_duration:.2f}s "
                f"but got {metadata.segment_info.duration_sec:.2f}s"
            )

        # Check at least one task is specified
        if not metadata.tasks_to_annotate:
            return False, "No tasks specified for annotation"

        return True, None
