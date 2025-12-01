"""Object tracking interfaces.

This module defines the interface for object tracking.
The actual implementation will be completed by the colleague
working on tracking model integration.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .bbox_annotator import BoundingBox

logger = logging.getLogger(__name__)


class TrackingResult:
    """Represents the tracking result for a video segment."""

    def __init__(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        bboxes: List[BoundingBox]
    ):
        """
        Initialize tracking result.

        Args:
            video_path: Path to the tracked video
            start_frame: Starting frame number
            end_frame: Ending frame number
            bboxes: List of bounding boxes, one for each frame
        """
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.bboxes = bboxes

        # Validate
        expected_frames = end_frame - start_frame + 1
        if len(bboxes) != expected_frames:
            logger.warning(
                f"Expected {expected_frames} bboxes but got {len(bboxes)}"
            )

    def get_bbox_at_frame(self, frame_number: int) -> Optional[BoundingBox]:
        """
        Get bounding box at a specific frame.

        Args:
            frame_number: Frame number (relative to video start)

        Returns:
            BoundingBox at that frame, or None if out of range
        """
        if frame_number < self.start_frame or frame_number > self.end_frame:
            return None

        index = frame_number - self.start_frame
        return self.bboxes[index]

    def to_dict(self) -> dict:
        """Convert to dictionary format for JSON serialization."""
        return {
            "video_path": str(self.video_path),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "bboxes": [bbox.to_list() for bbox in self.bboxes]
        }


class ObjectTracker:
    """
    Interface for object tracking.

    This class provides methods for tracking objects across video frames.
    """

    def __init__(self, backend: str = "local"):
        """
        Initialize object tracker.

        Args:
            backend: Tracking backend to use
                    - "local": Local tracking model
                    - "api": Remote API-based tracking
        """
        self.backend = backend
        logger.info(f"Initialized ObjectTracker with backend: {backend}")

    def track_from_first_bbox(
        self,
        video_path: Path,
        first_bbox: BoundingBox,
        start_frame: int,
        end_frame: int
    ) -> TrackingResult:
        """
        Track an object starting from its first bounding box.

        Args:
            video_path: Path to video file
            first_bbox: Bounding box of the object in the start_frame
            start_frame: Frame number where tracking starts
            end_frame: Frame number where tracking ends

        Returns:
            TrackingResult containing bboxes for all frames

        Raises:
            NotImplementedError: This method needs to be implemented

        Example:
            >>> tracker = ObjectTracker()
            >>> first_bbox = BoundingBox(100, 200, 200, 400)
            >>> result = tracker.track_from_first_bbox(
            ...     video_path=Path("game.mp4"),
            ...     first_bbox=first_bbox,
            ...     start_frame=10,
            ...     end_frame=50
            ... )
            >>> print(f"Tracked {len(result.bboxes)} frames")
            Tracked 41 frames
        """
        raise NotImplementedError(
            "track_from_first_bbox() needs to be implemented. "
            "This should use a tracking model (e.g., ByteTrack, DeepSORT, "
            "or a custom tracker) to generate bboxes for all frames."
        )

    def track_with_query(
        self,
        video_path: Path,
        query: str,
        start_frame: int,
        end_frame: int
    ) -> TrackingResult:
        """
        Track an object using a natural language query.

        This method first detects the object in the start_frame using
        the query, then tracks it through subsequent frames.

        Args:
            video_path: Path to video file
            query: Natural language query describing the object
                  (e.g., "track the player wearing red jersey number 10")
            start_frame: Frame number where tracking starts
            end_frame: Frame number where tracking ends

        Returns:
            TrackingResult containing bboxes for all frames

        Raises:
            NotImplementedError: This method needs to be implemented

        Example:
            >>> tracker = ObjectTracker()
            >>> result = tracker.track_with_query(
            ...     video_path=Path("game.mp4"),
            ...     query="the player wearing red jersey running forward",
            ...     start_frame=10,
            ...     end_frame=50
            ... )
            >>> print(result.get_bbox_at_frame(25))
            BoundingBox(xtl=150, ytl=220, xbr=250, ybr=420)
        """
        raise NotImplementedError(
            "track_with_query() needs to be implemented. "
            "This should first detect the object using the query, "
            "then track it across frames."
        )

    def track_multiple_objects(
        self,
        video_path: Path,
        first_bboxes: List[BoundingBox],
        start_frame: int,
        end_frame: int
    ) -> List[TrackingResult]:
        """
        Track multiple objects simultaneously.

        Args:
            video_path: Path to video file
            first_bboxes: List of bounding boxes for each object in start_frame
            start_frame: Frame number where tracking starts
            end_frame: Frame number where tracking ends

        Returns:
            List of TrackingResult, one for each object

        Raises:
            NotImplementedError: This method needs to be implemented

        Note:
            This is useful for tasks that need to track multiple players
            or objects at the same time.
        """
        raise NotImplementedError(
            "track_multiple_objects() needs to be implemented. "
            "This should track multiple objects and maintain their identities."
        )

    def validate_tracking_result(
        self,
        result: TrackingResult
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a tracking result.

        Args:
            result: TrackingResult to validate

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            >>> is_valid, error = tracker.validate_tracking_result(result)
            >>> if not is_valid:
            ...     print(f"Tracking validation failed: {error}")
        """
        # Check frame count
        expected_frames = result.end_frame - result.start_frame + 1
        if len(result.bboxes) != expected_frames:
            return False, (
                f"Expected {expected_frames} bboxes but got {len(result.bboxes)}"
            )

        # Check all bboxes are valid
        for i, bbox in enumerate(result.bboxes):
            if bbox.xtl >= bbox.xbr or bbox.ytl >= bbox.ybr:
                return False, f"Invalid bbox at frame {result.start_frame + i}"

        return True, None


# Example usage documentation
"""
Example usage for your colleague to implement:

```python
from auto_annotator.annotators.tracker import ObjectTracker, TrackingResult
from auto_annotator.annotators.bbox_annotator import BoundingBox
from pathlib import Path

# Initialize tracker
tracker = ObjectTracker(backend="local")

# Method 1: Track from first bounding box
video_path = Path("game.mp4")
first_bbox = BoundingBox(100, 200, 200, 400)
result = tracker.track_from_first_bbox(
    video_path=video_path,
    first_bbox=first_bbox,
    start_frame=10,
    end_frame=50
)

# Access tracking results
for frame_num in range(result.start_frame, result.end_frame + 1):
    bbox = result.get_bbox_at_frame(frame_num)
    print(f"Frame {frame_num}: {bbox.to_list()}")

# Method 2: Track with natural language query
result = tracker.track_with_query(
    video_path=video_path,
    query="the player wearing red jersey number 10 running forward",
    start_frame=10,
    end_frame=50
)

# Validate results
is_valid, error = tracker.validate_tracking_result(result)
if not is_valid:
    print(f"Validation failed: {error}")

# Export to JSON
tracking_data = result.to_dict()
```

Implementation notes:
1. Consider using existing tracking libraries (e.g., ByteTrack, DeepSORT)
2. Handle occlusions and tracking failures gracefully
3. Add confidence scores for each tracked bbox
4. Consider implementing tracking quality metrics
5. Add proper error handling and logging
6. For local backend, you might want to download and cache model weights
"""
