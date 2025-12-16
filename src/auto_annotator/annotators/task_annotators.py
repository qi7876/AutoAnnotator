"""Concrete implementations of task annotators."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..adapters import ClipMetadata
from ..utils import PromptLoader, VideoUtils
from .base_annotator import BaseAnnotator
from .bbox_annotator import BBoxAnnotator
from .gemini_client import GeminiClient
from .tracker import ObjectTracker

logger = logging.getLogger(__name__)


class ScoreboardSingleAnnotator(BaseAnnotator):
    """Annotator for Scoreboard Understanding - Single Frame task."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_loader: PromptLoader,
        bbox_annotator: BBoxAnnotator
    ):
        super().__init__(gemini_client, prompt_loader)
        self.bbox_annotator = bbox_annotator

    def get_task_name(self) -> str:
        return "ScoreboardSingle"

    def get_task_l1(self) -> str:
        return "Understanding"

    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate scoreboard in a single frame."""
        logger.info(f"Annotating {self.task_name} for {segment_metadata.segment_id}")

        # Upload video
        video_file = self.gemini_client.upload_video(
            segment_metadata.get_video_path(dataset_root)
        )

        try:
            # Load prompt
            prompt = self.load_prompt(segment_metadata)

            # Get annotation from AI
            result = self.gemini_client.annotate_video(video_file, prompt)

            # Extract bounding box description
            bbox_description = result.get("bounding_box", "")

            if isinstance(bbox_description, str):
                # Extract frame from video
                timestamp_frame = result.get("timestamp_frame", 0)
                frame_path = VideoUtils.extract_frame(
                    segment_metadata.get_video_path(dataset_root),
                    timestamp_frame
                )

                # Use bbox_annotator to generate actual bounding box
                bbox = self.bbox_annotator.annotate_single_object(
                    frame_path, bbox_description
                )
                result["bounding_box"] = bbox.to_list()
                result.setdefault("_debug", {})["frame_path"] = str(frame_path)

            # Add metadata
            result = self.add_metadata_fields(result)

            logger.info(f"Successfully annotated {self.task_name}")
            return result

        finally:
            # Cleanup uploaded file
            self.gemini_client.cleanup_file(video_file)


class ScoreboardMultipleAnnotator(BaseAnnotator):
    """Annotator for Scoreboard Understanding - Multiple Frames task."""

    def get_task_name(self) -> str:
        return "ScoreboardMultiple"

    def get_task_l1(self) -> str:
        return "Understanding"

    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate scoreboard changes across multiple frames."""
        logger.info(f"Annotating {self.task_name} for {segment_metadata.segment_id}")

        # Upload video
        video_file = self.gemini_client.upload_video(
            segment_metadata.get_video_path(dataset_root)
        )

        try:
            # Load prompt
            prompt = self.load_prompt(segment_metadata)

            # Get annotation from AI
            result = self.gemini_client.annotate_video(video_file, prompt)

            # Add metadata
            result = self.add_metadata_fields(result)

            logger.info(f"Successfully annotated {self.task_name}")
            return result

        finally:
            self.gemini_client.cleanup_file(video_file)


class ObjectsSpatialRelationshipsAnnotator(BaseAnnotator):
    """Annotator for Objects Spatial Relationships task."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_loader: PromptLoader,
        bbox_annotator: BBoxAnnotator
    ):
        super().__init__(gemini_client, prompt_loader)
        self.bbox_annotator = bbox_annotator

    def get_task_name(self) -> str:
        return "Objects_Spatial_Relationships"

    def get_task_l1(self) -> str:
        return "Understanding"

    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate spatial relationships between objects."""
        logger.info(f"Annotating {self.task_name} for {segment_metadata.segment_id}")

        # Upload video
        video_file = self.gemini_client.upload_video(
            segment_metadata.get_video_path(dataset_root)
        )

        try:
            # Load prompt
            prompt = self.load_prompt(segment_metadata)

            # Get annotation from AI
            result = self.gemini_client.annotate_video(video_file, prompt)

            # Process bounding boxes
            bbox_info = result.get("bounding_box", [])

            if isinstance(bbox_info, list) and bbox_info:
                # Extract frame
                timestamp_frame = result.get("timestamp_frame", 0)
                frame_path = VideoUtils.extract_frame(
                    segment_metadata.get_video_path(dataset_root),
                    timestamp_frame
                )

                # TODO: Use bbox_annotator to generate actual bounding boxes
                # descriptions = [obj["description"] for obj in bbox_info]
                # bboxes = self.bbox_annotator.annotate_multiple_objects(
                #     frame_path, descriptions
                # )
                #
                # for i, bbox in enumerate(bboxes):
                #     bbox_info[i]["box"] = bbox.to_list()

                logger.warning(
                    "Bounding box annotation not implemented. "
                    "Keeping natural language descriptions."
                )

            # Add metadata
            result = self.add_metadata_fields(result)

            logger.info(f"Successfully annotated {self.task_name}")
            return result

        finally:
            self.gemini_client.cleanup_file(video_file)


class SpatialTemporalGroundingAnnotator(BaseAnnotator):
    """Annotator for Spatial-Temporal Grounding task."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_loader: PromptLoader,
        bbox_annotator: BBoxAnnotator,
        tracker: ObjectTracker
    ):
        super().__init__(gemini_client, prompt_loader)
        self.bbox_annotator = bbox_annotator
        self.tracker = tracker

    def get_task_name(self) -> str:
        return "Spatial_Temporal_Grounding"

    def get_task_l1(self) -> str:
        return "Understanding"

    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate spatial-temporal grounding."""
        logger.info(f"Annotating {self.task_name} for {segment_metadata.segment_id}")

        # Upload video
        video_file = self.gemini_client.upload_video(
            segment_metadata.get_video_path(dataset_root)
        )

        try:
            # Load prompt
            prompt = self.load_prompt(segment_metadata)

            # Get annotation from AI
            result = self.gemini_client.annotate_video(video_file, prompt)

            # Get first frame description
            first_frame_desc = result.get("first_frame_description", "")
            a_window = result.get("A_window_frame", [])

            if first_frame_desc and len(a_window) == 2:
                # Extract first frame of answer window
                first_frame = a_window[0]
                frame_path = VideoUtils.extract_frame(
                    segment_metadata.get_video_path(dataset_root),
                    first_frame
                )

                # TODO: Use bbox_annotator and tracker
                # first_bbox = self.bbox_annotator.annotate_single_object(
                #     frame_path, first_frame_desc
                # )
                #
                # tracking_result = self.tracker.track_from_first_bbox(
                #     segment_metadata.get_video_path(dataset_root),
                #     first_bbox,
                #     a_window[0],
                #     a_window[1]
                # )
                #
                # result["first_bounding_box"] = first_bbox.to_list()
                # result["tracking_bboxes"] = [
                #     bbox.to_list() for bbox in tracking_result.bboxes
                # ]

                logger.warning(
                    "Bounding box and tracking not implemented. "
                    "Keeping first frame description only."
                )

            # Add metadata
            result = self.add_metadata_fields(result)

            logger.info(f"Successfully annotated {self.task_name}")
            return result

        finally:
            self.gemini_client.cleanup_file(video_file)


class ContinuousActionsCaptionAnnotator(BaseAnnotator):
    """Annotator for Continuous Actions Caption task."""

    def get_task_name(self) -> str:
        return "Continuous_Actions_Caption"

    def get_task_l1(self) -> str:
        return "Understanding"

    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate continuous actions."""
        logger.info(f"Annotating {self.task_name} for {segment_metadata.segment_id}")

        # Upload video
        video_file = self.gemini_client.upload_video(
            segment_metadata.get_video_path(dataset_root)
        )

        try:
            # Load prompt
            prompt = self.load_prompt(segment_metadata)

            # Get annotation from AI
            result = self.gemini_client.annotate_video(video_file, prompt)

            # Validate answer window alignment
            a_windows = result.get("A_window_frame", [])
            answers = result.get("answer", [])

            if len(a_windows) != len(answers):
                logger.warning(
                    f"Answer window count ({len(a_windows)}) doesn't match "
                    f"answer count ({len(answers)})"
                )

            # Add metadata
            result = self.add_metadata_fields(result)

            logger.info(f"Successfully annotated {self.task_name}")
            return result

        finally:
            self.gemini_client.cleanup_file(video_file)


class ContinuousEventsCaptionAnnotator(BaseAnnotator):
    """Annotator for Continuous Events Caption task."""

    def get_task_name(self) -> str:
        return "Continuous_Events_Caption"

    def get_task_l1(self) -> str:
        return "Understanding"

    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate continuous events."""
        logger.info(f"Annotating {self.task_name} for {segment_metadata.segment_id}")

        # Upload video
        video_file = self.gemini_client.upload_video(
            segment_metadata.get_video_path(dataset_root)
        )

        try:
            # Load prompt
            prompt = self.load_prompt(segment_metadata)

            # Get annotation from AI
            result = self.gemini_client.annotate_video(video_file, prompt)

            # Validate answer window alignment
            a_windows = result.get("A_window_frame", [])
            answers = result.get("answer", [])

            if len(a_windows) != len(answers):
                logger.warning(
                    f"Answer window count ({len(a_windows)}) doesn't match "
                    f"answer count ({len(answers)})"
                )

            # Add metadata
            result = self.add_metadata_fields(result)

            logger.info(f"Successfully annotated {self.task_name}")
            return result

        finally:
            self.gemini_client.cleanup_file(video_file)


class ObjectTrackingAnnotator(BaseAnnotator):
    """Annotator for Object Tracking task."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        prompt_loader: PromptLoader,
        bbox_annotator: BBoxAnnotator,
        tracker: ObjectTracker
    ):
        super().__init__(gemini_client, prompt_loader)
        self.bbox_annotator = bbox_annotator
        self.tracker = tracker

    def get_task_name(self) -> str:
        return "Object_Tracking"

    def get_task_l1(self) -> str:
        return "Perception"

    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate object tracking."""
        logger.info(f"Annotating {self.task_name} for {segment_metadata.segment_id}")

        # Upload video
        video_file = self.gemini_client.upload_video(
            segment_metadata.get_video_path(dataset_root)
        )

        try:
            # Load prompt
            prompt = self.load_prompt(segment_metadata)

            # Get annotation from AI
            result = self.gemini_client.annotate_video(video_file, prompt)

            # Get first frame description and tracking window
            first_frame_desc = result.get("first_frame_description", "")
            q_window = result.get("Q_window_frame", [])

            if first_frame_desc and len(q_window) == 2:
                # Extract first frame
                first_frame = q_window[0]
                frame_path = VideoUtils.extract_frame(
                    segment_metadata.get_video_path(dataset_root),
                    first_frame
                )

                # TODO: Use bbox_annotator and tracker
                first_bboxes_with_label = []
                multi_object = len(first_frame_desc)>1
                if(len(first_frame_desc)==0):
                    raise ValueError("No description for first frame bounding box.")
                # single object
                elif(not multi_object):
                    first_bboxes = self.bbox_annotator.annotate_single_object(
                        frame_path, first_frame_desc[0]
                    )
                    bbox_dict = {}
                    bbox_dict['bbox'] = first_bboxes.to_list()
                    bbox_dict['label'] = first_frame_desc[0]
                    first_bboxes_with_label.append(bbox_dict)
                # multiple objects
                elif(multi_object):
                    first_bboxes = self.bbox_annotator.annotate_multiple_objects(
                        frame_path, first_frame_desc
                    )
                    for i, bbox in enumerate(first_bboxes):
                        bbox_dict = {}
                        bbox_dict['bbox'] = bbox.to_list()
                        bbox_dict['label'] = first_frame_desc[i]
                        first_bboxes_with_label.append(bbox_dict)
                
                tracking_result = self.tracker.track_from_first_bbox(
                    segment_metadata.get_video_path(),
                    first_bboxes_with_label,
                    q_window[0],
                    q_window[1]
                )
                # Handle tracking result (now returns TrackingResult object)
                result["first_bounding_box"] = first_bboxes_with_label if multi_object else first_bboxes.to_list()
                result["tracking_bboxes"] = tracking_result.to_dict()

            # Add metadata
            result = self.add_metadata_fields(result)

            logger.info(f"Successfully annotated {self.task_name}")
            return result

        finally:
            self.gemini_client.cleanup_file(video_file)


class TaskAnnotatorFactory:
    """Factory for creating task annotators."""

    @staticmethod
    def create_annotator(
        task_name: str,
        gemini_client: GeminiClient,
        prompt_loader: PromptLoader,
        bbox_annotator: BBoxAnnotator,
        tracker: ObjectTracker
    ) -> BaseAnnotator:
        """
        Create an annotator for a specific task.

        Args:
            task_name: Task name
            gemini_client: Gemini API client
            prompt_loader: Prompt loader
            bbox_annotator: Bounding box annotator
            tracker: Object tracker

        Returns:
            Task annotator instance

        Raises:
            ValueError: If task name is not recognized
        """
        annotators = {
            "ScoreboardSingle": lambda: ScoreboardSingleAnnotator(
                gemini_client, prompt_loader, bbox_annotator
            ),
            "ScoreboardMultiple": lambda: ScoreboardMultipleAnnotator(
                gemini_client, prompt_loader
            ),
            "Objects_Spatial_Relationships": lambda: ObjectsSpatialRelationshipsAnnotator(
                gemini_client, prompt_loader, bbox_annotator
            ),
            "Spatial_Temporal_Grounding": lambda: SpatialTemporalGroundingAnnotator(
                gemini_client, prompt_loader, bbox_annotator, tracker
            ),
            "Continuous_Actions_Caption": lambda: ContinuousActionsCaptionAnnotator(
                gemini_client, prompt_loader
            ),
            "Continuous_Events_Caption": lambda: ContinuousEventsCaptionAnnotator(
                gemini_client, prompt_loader
            ),
            "Object_Tracking": lambda: ObjectTrackingAnnotator(
                gemini_client, prompt_loader, bbox_annotator, tracker
            ),
        }

        if task_name not in annotators:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {list(annotators.keys())}"
            )

        return annotators[task_name]()

    @staticmethod
    def get_available_tasks() -> list[str]:
        """Get list of all available task names."""
        return [
            "ScoreboardSingle",
            "ScoreboardMultiple",
            "Objects_Spatial_Relationships",
            "Spatial_Temporal_Grounding",
            "Continuous_Actions_Caption",
            "Continuous_Events_Caption",
            "Object_Tracking",
        ]
