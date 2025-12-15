"""Input adapter for handling segment metadata from previous steps."""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class OriginalVideoInfo(BaseModel):
    """Information about the original video."""

    sport: str
    event: str

    def get_video_path(self, dataset_root: Optional[Path] = None, video_id: int = 1) -> Path:
        """
        Construct the path to the original video file.

        Args:
            dataset_root: Root directory of the dataset. If None, uses "Dataset"
            video_id: Video ID (default: 1, can be extracted from segment_id)

        Returns:
            Path to the original video file (e.g., 1.mp4, 2.mp4, etc.)
        """
        if dataset_root is None:
            dataset_root = Path("Dataset")

        return dataset_root / self.sport / self.event / f"{video_id}.mp4"

    def get_json_path(self, dataset_root: Optional[Path] = None, video_id: int = 1) -> Path:
        """
        Construct the path to the original video's JSON metadata file.

        Args:
            dataset_root: Root directory of the dataset. If None, uses "Dataset"
            video_id: Video ID (default: 1)

        Returns:
            Path to the original video's JSON file (e.g., 1.json, 2.json, etc.)
        """
        if dataset_root is None:
            dataset_root = Path("Dataset")

        return dataset_root / self.sport / self.event / f"{video_id}.json"

    def get_metainfo_path(self, dataset_root: Optional[Path] = None) -> Path:
        """
        Construct the path to the event's metainfo.json file.

        Args:
            dataset_root: Root directory of the dataset. If None, uses "Dataset"

        Returns:
            Path to the metainfo.json file
        """
        if dataset_root is None:
            dataset_root = Path("Dataset")

        return dataset_root / self.sport / self.event / "metainfo.json"


class SegmentInfo(BaseModel):
    """Information about a video segment or single frame."""

    start_frame_in_original: int = Field(..., description="Starting frame number in the original video")
    total_frames: int = Field(..., description="Total number of frames (1 for single frames)")
    fps: float = Field(..., description="Frames per second")
    duration_sec: float = Field(..., description="Duration in seconds")
    resolution: List[int] = Field(..., description="Video resolution [width, height]")

    def is_single_frame(self) -> bool:
        """Check if this is a single frame (total_frames == 1)."""
        return self.total_frames == 1

    def is_segment(self) -> bool:
        """Check if this is a video segment (total_frames > 1)."""
        return self.total_frames > 1


class SegmentMetadata(BaseModel):
    """Complete metadata for a video segment or single frame."""

    segment_id: Union[int, str] = Field(
        ..., description="Unique identifier (int for single frames, string like '1_split_1_start_000292' for segments)"
    )
    original_video: OriginalVideoInfo
    segment_info: SegmentInfo
    tasks_to_annotate: List[str] = Field(
        default_factory=list, description="List of annotation tasks to perform"
    )
    additional_info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def has_task(self, task_name: str) -> bool:
        """Check if a task should be annotated for this segment."""
        return task_name in self.tasks_to_annotate

    def _parse_segment_id(self) -> Dict[str, Any]:
        """
        Parse segment_id to extract video_id and other information.

        Returns:
            Dict with keys: video_id, split_num (optional), start_frame (optional)
        """
        if isinstance(self.segment_id, int):
            # For single frames with integer segment_id, assume video_id = 1
            return {"video_id": 1, "is_single_frame": True}

        # Try to parse segment_id string format: "{video_id}_split_{split_num}_start_{start_frame}"
        match = re.match(r"(\d+)_split_(\d+)_start_(\d+)", str(self.segment_id))
        if match:
            return {
                "video_id": int(match.group(1)),
                "split_num": int(match.group(2)),
                "start_frame": int(match.group(3)),
                "is_single_frame": False
            }

        # Try to parse single frame format: "{video_id}_frame_{frame_num}"
        match = re.match(r"(\d+)_frame_(\d+)", str(self.segment_id))
        if match:
            return {
                "video_id": int(match.group(1)),
                "frame_num": int(match.group(2)),
                "is_single_frame": True
            }

        # Fallback: treat as video_id = 1
        return {"video_id": 1, "is_single_frame": self.segment_info.is_single_frame()}

    def get_video_path(self, dataset_root: Optional[Path] = None) -> Path:
        """
        Get the path to the segment video or single frame image.

        For segments: Dataset/{sport}/{event}/segment_dir/{segment_id}.mp4
        For single frames: Dataset/{sport}/{event}/singleframes_dir/{segment_id}.jpg

        Args:
            dataset_root: Root directory of the dataset. If None, uses "Dataset"

        Returns:
            Path to the segment video file or single frame image
        """
        if dataset_root is None:
            dataset_root = Path("Dataset")

        base_path = dataset_root / self.original_video.sport / self.original_video.event

        if self.segment_info.is_single_frame():
            # Single frame: singleframes_dir/{segment_id}.jpg
            return base_path / "singleframes_dir" / f"{self.segment_id}.jpg"
        else:
            # Video segment: segment_dir/{segment_id}.mp4
            return base_path / "segment_dir" / f"{self.segment_id}.mp4"

    def get_json_path(self, dataset_root: Optional[Path] = None) -> Path:
        """
        Get the path to the segment's JSON metadata file.

        For segments: Dataset/{sport}/{event}/segment_dir/{segment_id}.json
        For single frames: Dataset/{sport}/{event}/singleframes_dir/{segment_id}.json

        Args:
            dataset_root: Root directory of the dataset. If None, uses "Dataset"

        Returns:
            Path to the segment's JSON metadata file
        """
        if dataset_root is None:
            dataset_root = Path("Dataset")

        base_path = dataset_root / self.original_video.sport / self.original_video.event

        if self.segment_info.is_single_frame():
            # Single frame: singleframes_dir/{segment_id}.json
            return base_path / "singleframes_dir" / f"{self.segment_id}.json"
        else:
            # Video segment: segment_dir/{segment_id}.json
            return base_path / "segment_dir" / f"{self.segment_id}.json"

    def get_original_video_path(self, dataset_root: Optional[Path] = None) -> Path:
        """
        Get the path to the original video file.

        Args:
            dataset_root: Root directory of the dataset. If None, uses "Dataset"

        Returns:
            Path to the original video file
        """
        parsed = self._parse_segment_id()
        video_id = parsed.get("video_id", 1)
        return self.original_video.get_video_path(dataset_root, video_id=video_id)


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
    def load_from_directory(
        segments_dir: Path,
        single_frame_only: bool = False
    ) -> List[SegmentMetadata]:
        """
        Load all segment metadata files from a directory.

        Args:
            segments_dir: Directory containing segment metadata JSON files
                         (e.g., segment_dir or singleframes_dir)
            single_frame_only: If True, only load single frame metadata (total_frames == 1)

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

                # Filter by type if specified
                if single_frame_only and not metadata.segment_info.is_single_frame():
                    continue

                metadata_list.append(metadata)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue

        return metadata_list

    @staticmethod
    def load_from_event_directory(
        event_dir: Path,
        single_frame_only: bool = False
    ) -> List[SegmentMetadata]:
        """
        Load all segment metadata from an event directory (both segment_dir and singleframes_dir).

        Args:
            event_dir: Event directory (e.g., Dataset/Archery/Men's_Individual)
            single_frame_only: If True, only load single frame metadata

        Returns:
            List of SegmentMetadata objects
        """
        metadata_list = []

        # Load from segment_dir
        segment_dir = event_dir / "segment_dir"
        if segment_dir.exists() and not single_frame_only:
            metadata_list.extend(
                InputAdapter.load_from_directory(segment_dir, single_frame_only=False)
            )

        # Load from singleframes_dir
        singleframes_dir = event_dir / "singleframes_dir"
        if singleframes_dir.exists():
            metadata_list.extend(
                InputAdapter.load_from_directory(singleframes_dir, single_frame_only=True)
            )

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
    def validate_metadata(
        metadata: SegmentMetadata,
        dataset_root: Optional[Path] = None,
        check_file_existence: bool = True
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that segment metadata is complete and consistent.

        Args:
            metadata: SegmentMetadata to validate
            dataset_root: Root directory of the dataset
            check_file_existence: Whether to check if video files exist

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check video/image file exists (optional)
        if check_file_existence:
            video_path = metadata.get_video_path(dataset_root)
            if not video_path.exists():
                return False, f"Content file not found: {video_path}"

            # Check original video file exists
            original_path = metadata.get_original_video_path(dataset_root)
            if not original_path.exists():
                return False, f"Original video file not found: {original_path}"

        # Check start_frame is valid
        if metadata.segment_info.start_frame_in_original < 0:
            return False, "start_frame_in_original must be non-negative"

        # Check total_frames is valid
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

        # Check at least one task is specified (allow empty string in tasks list)
        if not metadata.tasks_to_annotate:
            return False, "No tasks specified for annotation"

        return True, None
