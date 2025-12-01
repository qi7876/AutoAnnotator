"""Video processing utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class VideoUtils:
    """Utilities for video processing."""

    @staticmethod
    def get_video_info(video_path: Path) -> dict:
        """
        Get video metadata information.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary containing video metadata

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file cannot be opened
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            info = {
                "fps": fps,
                "total_frames": frame_count,
                "resolution": [width, height],
                "duration_sec": round(duration, 2)
            }

            logger.info(f"Video info for {video_path.name}: {info}")
            return info

        finally:
            cap.release()

    @staticmethod
    def extract_frame(
        video_path: Path,
        frame_number: int,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Extract a specific frame from video.

        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (0-indexed)
            output_path: Optional output path for frame image

        Returns:
            Path to extracted frame image

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If frame number is invalid
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            # Check frame number validity
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_number < 0 or frame_number >= total_frames:
                raise ValueError(
                    f"Invalid frame number {frame_number}. "
                    f"Video has {total_frames} frames."
                )

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_number}")

            # Determine output path
            if output_path is None:
                output_path = (
                    video_path.parent /
                    f"{video_path.stem}_frame_{frame_number:06d}.jpg"
                )

            # Save frame
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame)

            logger.info(f"Extracted frame {frame_number} to {output_path}")
            return output_path

        finally:
            cap.release()

    @staticmethod
    def frames_to_seconds(frames: int, fps: int) -> float:
        """
        Convert frame count to seconds.

        Args:
            frames: Number of frames
            fps: Frames per second

        Returns:
            Time in seconds
        """
        if fps <= 0:
            raise ValueError("FPS must be positive")
        return frames / fps

    @staticmethod
    def seconds_to_frames(seconds: float, fps: int) -> int:
        """
        Convert seconds to frame count.

        Args:
            seconds: Time in seconds
            fps: Frames per second

        Returns:
            Number of frames
        """
        if fps <= 0:
            raise ValueError("FPS must be positive")
        return int(seconds * fps)

    @staticmethod
    def validate_video_file(video_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate that a video file can be opened and read.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not video_path.exists():
            return False, f"Video file not found: {video_path}"

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            cap.release()
            return False, f"Cannot open video file: {video_path}"

        # Try to read first frame
        ret, _ = cap.read()
        cap.release()

        if not ret:
            return False, "Cannot read frames from video"

        return True, None
