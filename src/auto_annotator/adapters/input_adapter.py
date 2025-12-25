"""Input adapter for handling clip metadata from previous steps."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field


class OriginInfo(BaseModel):
    """Information about the original video source."""

    sport: str
    event: str

    def get_video_path(self, dataset_root: Optional[Path] = None, video_id: str = "1") -> Path:
        """
        Construct the path to the original video file.

        Args:
            dataset_root: Root directory of the dataset. If None, uses "data/Dataset"
            video_id: Video ID (default: "1")

        Returns:
            Path to the original video file (e.g., 1.mp4, 2.mp4, etc.)
        """
        if dataset_root is None:
            dataset_root = Path("data") / "Dataset"

        return dataset_root / self.sport / self.event / f"{video_id}.mp4"

    def get_json_path(self, dataset_root: Optional[Path] = None, video_id: str = "1") -> Path:
        """
        Construct the path to the original video's JSON metadata file.

        Args:
            dataset_root: Root directory of the dataset. If None, uses "data/Dataset"
            video_id: Video ID (default: "1")

        Returns:
            Path to the original video's JSON file (e.g., 1.json, 2.json, etc.)
        """
        if dataset_root is None:
            dataset_root = Path("data") / "Dataset"

        return dataset_root / self.sport / self.event / f"{video_id}.json"

    def get_metainfo_path(self, dataset_root: Optional[Path] = None) -> Path:
        """
        Construct the path to the event's metainfo.json file.

        Args:
            dataset_root: Root directory of the dataset. If None, uses "data/Dataset"

        Returns:
            Path to the metainfo.json file
        """
        if dataset_root is None:
            dataset_root = Path("data") / "Dataset"

        return dataset_root / self.sport / self.event / "metainfo.json"


class ClipInfo(BaseModel):
    """Information about a video clip or single frame."""

    original_starting_frame: int = Field(..., description="Starting frame number in the original video")
    total_frames: int = Field(..., description="Total number of frames (1 for single frames)")
    fps: float = Field(..., description="Frames per second")

    def is_single_frame(self) -> bool:
        """Check if this is a single frame (total_frames == 1)."""
        return self.total_frames == 1

    def is_clip(self) -> bool:
        """Check if this is a video clip (total_frames > 1)."""
        return self.total_frames > 1


class ClipMetadata(BaseModel):
    """Complete metadata for a video clip or single frame."""

    id: str = Field(..., description="Unique identifier matching the filename")
    origin: OriginInfo
    info: ClipInfo
    tasks_to_annotate: List[str] = Field(
        default_factory=list, description="List of annotation tasks to perform"
    )

    def has_task(self, task_name: str) -> bool:
        """Check if a task should be annotated for this clip."""
        return task_name in self.tasks_to_annotate

    def get_video_path(self, dataset_root: Optional[Path] = None) -> Path:
        """
        Get the path to the clip video or single frame image.

        For clips: data/Dataset/{sport}/{event}/clips/{id}.mp4
        For single frames: data/Dataset/{sport}/{event}/frames/{id}.jpg

        Args:
            dataset_root: Root directory of the dataset. If None, uses "data/Dataset"

        Returns:
            Path to the clip video file or single frame image
        """
        if dataset_root is None:
            dataset_root = Path("data") / "Dataset"

        base_path = dataset_root / self.origin.sport / self.origin.event

        if self.info.is_single_frame():
            # Single frame: frames/{id}.jpg
            return base_path / "frames" / f"{self.id}.jpg"
        else:
            # Video clip: clips/{id}.mp4
            return base_path / "clips" / f"{self.id}.mp4"

    def get_json_path(self, dataset_root: Optional[Path] = None) -> Path:
        """
        Get the path to the clip's JSON metadata file.

        For clips: data/Dataset/{sport}/{event}/clips/{id}.json
        For single frames: data/Dataset/{sport}/{event}/frames/{id}.json

        Args:
            dataset_root: Root directory of the dataset. If None, uses "data/Dataset"

        Returns:
            Path to the clip's JSON metadata file
        """
        if dataset_root is None:
            dataset_root = Path("data") / "Dataset"

        base_path = dataset_root / self.origin.sport / self.origin.event

        if self.info.is_single_frame():
            # Single frame: frames/{id}.json
            return base_path / "frames" / f"{self.id}.json"
        else:
            # Video clip: clips/{id}.json
            return base_path / "clips" / f"{self.id}.json"

    def get_original_video_path(self, dataset_root: Optional[Path] = None, video_id: Optional[str] = None) -> Path:
        """
        Get the path to the original video file.

        Args:
            dataset_root: Root directory of the dataset. If None, uses "data/Dataset"
            video_id: Video ID (if None, defaults to "1")

        Returns:
            Path to the original video file
        """
        if video_id is None:
            video_id = "1"
        return self.origin.get_video_path(dataset_root, video_id=video_id)


class InputAdapter:
    """
    Adapter for handling different input formats from previous steps.

    This class provides a unified interface for loading clip metadata,
    regardless of the format used by the clip splitting step (step 1-2).
    """

    @staticmethod
    def load_from_json(json_path: Path) -> ClipMetadata:
        """
        Load clip metadata from a JSON file.

        Args:
            json_path: Path to the clip metadata JSON file

        Returns:
            ClipMetadata object

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON format is invalid
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Clip metadata file not found: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ClipMetadata(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading clip metadata: {e}")

    @staticmethod
    def load_from_directory(
        clips_dir: Path,
        single_frame_only: bool = False
    ) -> List[ClipMetadata]:
        """
        Load all clip metadata files from a directory.

        Args:
            clips_dir: Directory containing clip metadata JSON files
                       (e.g., clips or frames)
            single_frame_only: If True, only load single frame metadata (total_frames == 1)

        Returns:
            List of ClipMetadata objects
        """
        if not clips_dir.exists():
            raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

        metadata_list = []
        for json_file in clips_dir.glob("*.json"):
            # Skip if this is not a clip metadata file
            # (e.g., might be an annotation result file)
            if json_file.stem.startswith("annotation_"):
                continue

            try:
                metadata = InputAdapter.load_from_json(json_file)

                # Filter by type if specified
                if single_frame_only and not metadata.info.is_single_frame():
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
    ) -> List[ClipMetadata]:
        """
        Load all clip metadata from an event directory (both clips and frames).

        Args:
            event_dir: Event directory (e.g., data/Dataset/Archery/Men's_Individual)
            single_frame_only: If True, only load single frame metadata

        Returns:
            List of ClipMetadata objects
        """
        metadata_list = []

        # Load from clips directory
        clips_dir = event_dir / "clips"
        if clips_dir.exists() and not single_frame_only:
            metadata_list.extend(
                InputAdapter.load_from_directory(clips_dir, single_frame_only=False)
            )

        # Load from frames directory
        frames_dir = event_dir / "frames"
        if frames_dir.exists():
            metadata_list.extend(
                InputAdapter.load_from_directory(frames_dir, single_frame_only=True)
            )

        return metadata_list

    @staticmethod
    def create_from_dict(data: dict) -> ClipMetadata:
        """
        Create ClipMetadata from a dictionary.

        This method is useful for testing or when receiving data
        from other Python components.

        Args:
            data: Dictionary containing clip metadata

        Returns:
            ClipMetadata object
        """
        return ClipMetadata(**data)

    @staticmethod
    def validate_metadata(
        metadata: ClipMetadata,
        dataset_root: Optional[Path] = None,
        check_file_existence: bool = True
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that clip metadata is complete and consistent.

        Args:
            metadata: ClipMetadata to validate
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

        # Check original_starting_frame is valid
        if metadata.info.original_starting_frame < 0:
            return False, "original_starting_frame must be non-negative"

        # Check total_frames is valid
        if metadata.info.total_frames <= 0:
            return False, "total_frames must be positive"

        # Check FPS is valid
        if metadata.info.fps <= 0:
            return False, "fps must be positive"

        # Check at least one task is specified (allow empty string in tasks list)
        if not metadata.tasks_to_annotate:
            return False, "No tasks specified for annotation"

        return True, None
