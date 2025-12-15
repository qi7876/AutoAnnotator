"""Object tracking interfaces.

This module defines the interface for object tracking.
The actual implementation will be completed by the colleague
working on tracking model integration.
"""

import logging
import os
import sys
import cv2
import torch
import gc
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from .bbox_annotator import BoundingBox

# Add sam2 model path
samurai_path = Path(__file__).parent / "samurai_deploy"
sam2_path = samurai_path / "sam2"

# Add the samurai_deploy directory to sys.path so sam2 can be imported as a module
if str(samurai_path) not in sys.path:
    sys.path.insert(0, str(samurai_path))

try:
    # Change working directory temporarily to samurai_deploy for proper sam2 import
    original_cwd = os.getcwd()
    os.chdir(str(samurai_path))
    
    # Import sam2 from the proper location
    import sam2.build_sam as build_sam_module
    build_sam2_video_predictor = build_sam_module.build_sam2_video_predictor
    
    # Restore original working directory
    os.chdir(original_cwd)
    
except Exception as e:
    # This might happen during static analysis, but should work at runtime
    build_sam2_video_predictor = None
    # Restore original working directory if changed
    try:
        os.chdir(original_cwd)
    except:
        pass

logger = logging.getLogger(__name__)


class TrackingResult:
    """Represents the tracking result for a video segment with multiple objects."""

    def __init__(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        objects: List[dict]
    ):
        """
        Initialize tracking result.

        Args:
            video_path: Path to the tracked video
            start_frame: Starting frame number
            end_frame: Ending frame number
            objects: List of object tracking data, each containing id, label, and frames
        """
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.objects = objects

        # Validate
        expected_frames = end_frame - start_frame + 1
        for obj in objects:
            if len(obj.get('frames', {})) != expected_frames:
                logger.warning(
                    f"Object {obj.get('id')} expected {expected_frames} frames but got {len(obj.get('frames', {}))}"
                )

    def get_bbox_at_frame(self, frame_number: int, object_id: int = 0) -> Optional[List[float]]:
        """
        Get bounding box at a specific frame for a specific object.

        Args:
            frame_number: Frame number (relative to video start)
            object_id: Object ID to get bbox for (default 0 for first object)

        Returns:
            Bbox list [xtl, ytl, xbr, ybr] at that frame, or None if out of range
        """
        if frame_number < self.start_frame or frame_number > self.end_frame:
            return None

        for obj in self.objects:
            if obj.get('id') == object_id:
                frames = obj.get('frames', {})
                # Try both int and str keys for backward compatibility
                return frames.get(frame_number) or frames.get(str(frame_number))
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary format for JSON serialization."""
        # Convert frame indices to strings for JSON compatibility
        objects_serializable = []
        for obj in self.objects:
            obj_copy = obj.copy()
            if 'frames' in obj_copy:
                # Convert integer frame indices to strings
                obj_copy['frames'] = {str(k): v for k, v in obj_copy['frames'].items()}
            objects_serializable.append(obj_copy)
        
        return {
            "video_path": str(self.video_path),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "objects": objects_serializable
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
        self.samurai_path = Path(__file__).parent / "samurai_deploy"
        self.model_path = self.samurai_path / "sam2.1_hiera_base_plus.pt"
        logger.info(f"Initialized ObjectTracker with backend: {backend}")
        logger.info(f"SAMURAI model path: {self.model_path}")

    def track_from_first_bbox(
        self,
        video_path: Path,
        first_bboxes_with_label: List[dict],
        start_frame: int,
        end_frame: int
    ) -> TrackingResult:
        """
        Track objects starting from their first bounding boxes.

        Args:
            video_path: Path to video file
            first_bboxes_with_label: List of bbox dictionaries with 'bbox' and 'label' keys
            start_frame: Frame number where tracking starts
            end_frame: Frame number where tracking ends

        Returns:
            TrackingResult containing all tracked objects

        Example:
            >>> tracker = ObjectTracker()
            >>> first_bboxes = [{'bbox': [100, 200, 200, 400], 'label': 'player'}]
            >>> result = tracker.track_from_first_bbox(
            ...     video_path=Path("game.mp4"),
            ...     first_bboxes_with_label=first_bboxes,
            ...     start_frame=10,
            ...     end_frame=50
            ... )
            >>> print(f"Tracked {len(result.objects)} objects")
            Tracked 1 objects
        """
        logger.info(f"Starting tracking from frame {start_frame} to {end_frame}")
        logger.info(f"Tracking {len(first_bboxes_with_label)} objects")
        
        if not first_bboxes_with_label:
            logger.warning("No objects to track")
            return TrackingResult(video_path, start_frame, end_frame, [])
        
        if build_sam2_video_predictor is None:
            logger.error("SAM2 model not available, falling back to static bboxes")
            # Return fallback results 
            fallback_results = []
            for obj_id, bbox_data in enumerate(first_bboxes_with_label):
                fallback_obj = {
                    'id': obj_id,
                    'label': bbox_data['label'],
                    'frames': {}
                }
                # Repeat first bbox for all frames
                for frame_idx in range(start_frame, end_frame + 1):
                    fallback_obj['frames'][frame_idx] = bbox_data['bbox']
                fallback_results.append(fallback_obj)
            return TrackingResult(video_path, start_frame, end_frame, fallback_results)
            
        tracking_results = []
        
        try:
            # Determine model configuration based on model file name
            model_cfg = self._determine_model_cfg(str(self.model_path))
            
            # Build SAM2 video predictor
            device = "cpu" if not torch.cuda.is_available() else "cuda"
            predictor = build_sam2_video_predictor(model_cfg, str(self.model_path), device=device)
            
            # Initialize video state
            state = predictor.init_state(str(video_path), offload_video_to_cpu=True)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                # Add bounding box prompts for each object at start_frame
                for obj_id, bbox_data in enumerate(first_bboxes_with_label):
                    bbox_list = bbox_data['bbox']
                    # Convert to SAMURAI format [xtl, ytl, xbr, ybr] if needed
                    bbox_prompt = bbox_list
                    
                    predictor.add_new_points_or_box(
                        state,
                        box=bbox_prompt,
                        frame_idx=start_frame,
                        obj_id=obj_id
                    )
                
                # Track through video frames
                frame_results = {}  # frame_idx -> list of object bboxes
                
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    # Only process frames within our range
                    if frame_idx < start_frame:
                        continue
                    if frame_idx > end_frame:
                        break
                        
                    frame_bboxes = []
                    
                    # Process each tracked object
                    for expected_obj_id in range(len(first_bboxes_with_label)):
                        if expected_obj_id in object_ids:
                            mask_idx = object_ids.index(expected_obj_id)
                            mask = masks[mask_idx][0].cpu().numpy()
                            mask = mask > 0.0
                            
                            # Convert mask to bounding box
                            bbox = self._mask_to_bbox(mask)
                            frame_bboxes.append({
                                'id': expected_obj_id,
                                'bbox': bbox.to_list(),
                                'label': first_bboxes_with_label[expected_obj_id]['label']
                            })
                        else:
                            # Object lost, use zero bbox
                            frame_bboxes.append({
                                'id': expected_obj_id,
                                'bbox': [0, 0, 0, 0],
                                'label': first_bboxes_with_label[expected_obj_id]['label']
                            })
                    
                    frame_results[frame_idx] = frame_bboxes
                
                # Format results per object
                for obj_id, bbox_data in enumerate(first_bboxes_with_label):
                    obj_tracking = {
                        'id': obj_id,
                        'label': bbox_data['label'],
                        'frames': {}
                    }
                    
                    for frame_idx in range(start_frame, end_frame + 1):
                        if frame_idx in frame_results:
                            obj_bbox = next((fb for fb in frame_results[frame_idx] if fb['id'] == obj_id), None)
                            if obj_bbox:
                                obj_tracking['frames'][frame_idx] = obj_bbox['bbox']
                            else:
                                obj_tracking['frames'][frame_idx] = [0, 0, 0, 0]
                        else:
                            obj_tracking['frames'][frame_idx] = [0, 0, 0, 0]
                    
                    tracking_results.append(obj_tracking)
            
            # Clean up resources
            del predictor, state
            gc.collect()
            torch.clear_autocast_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"Successfully tracked {len(tracking_results)} objects across {end_frame - start_frame + 1} frames")
            return TrackingResult(video_path, start_frame, end_frame, tracking_results)
            
        except Exception as e:
            logger.error(f"Tracking failed: {e}")
            # Return fallback results 
            fallback_results = []
            for obj_id, bbox_data in enumerate(first_bboxes_with_label):
                fallback_obj = {
                    'id': obj_id,
                    'label': bbox_data['label'],
                    'frames': {}
                }
                # Repeat first bbox for all frames
                for frame_idx in range(start_frame, end_frame + 1):
                    fallback_obj['frames'][frame_idx] = bbox_data['bbox']
                fallback_results.append(fallback_obj)
            
            return TrackingResult(video_path, start_frame, end_frame, fallback_results)
    
    def _determine_model_cfg(self, model_path: str) -> str:
        """
        Determine model configuration file based on model path.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Path to configuration file
        """
        # Return full path to configuration file as used in demo.py
        if "large" in model_path:
            return "configs/samurai/sam2.1_hiera_l.yaml"
        elif "base_plus" in model_path:
            return "configs/samurai/sam2.1_hiera_b+.yaml"
        elif "small" in model_path:
            return "configs/samurai/sam2.1_hiera_s.yaml"
        elif "tiny" in model_path:
            return "configs/samurai/sam2.1_hiera_t.yaml"
        else:
            # Default to base_plus
            return "configs/samurai/sam2.1_hiera_b+.yaml"
    
    def _mask_to_bbox(self, mask: np.ndarray) -> BoundingBox:
        """
        Convert binary mask to bounding box.
        
        Args:
            mask: Binary mask array
            
        Returns:
            BoundingBox object
        """
        non_zero = np.argwhere(mask)
        
        if len(non_zero) == 0:
            # Empty mask, return zero bbox
            return BoundingBox(0, 0, 0, 0)
        else:
            y_min, x_min = non_zero.min(axis=0).tolist()
            y_max, x_max = non_zero.max(axis=0).tolist()
            return BoundingBox(float(x_min), float(y_min), float(x_max), float(y_max))

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
