"""Bounding box annotation interfaces.

This module defines the interface for bounding box annotation.
The actual implementation will be completed by the colleague
working on bounding box detection and grounding.
"""

import logging
from . import gemini_client
from pathlib import Path
from typing import List, Tuple, Union, Optional


from PIL import Image

logger = logging.getLogger(__name__)


class BoundingBox:
    """Represents a bounding box."""

    def __init__(self, xtl: float, ytl: float, xbr: float, ybr: float):
        """
        Initialize bounding box.

        Args:
            xtl: Top-left x coordinate
            ytl: Top-left y coordinate
            xbr: Bottom-right x coordinate
            ybr: Bottom-right y coordinate

        Note:
            Coordinates should be in pixel values, not normalized.
        """
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr

    def to_list(self) -> List[float]:
        """Convert to list format [xtl, ytl, xbr, ybr]."""
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "xtl": self.xtl,
            "ytl": self.ytl,
            "xbr": self.xbr,
            "ybr": self.ybr
        }

    @classmethod
    def from_normalized(
        cls,
        ymin_norm: float,
        xmin_norm: float,
        ymax_norm: float,
        xmax_norm: float,
        image_width: int,
        image_height: int
    ):
        """
        Create bounding box from normalized coordinates [0, 1000].

        Args:
            ymin_norm: Normalized top y (0-1000)
            xmin_norm: Normalized left x (0-1000)
            ymax_norm: Normalized bottom y (0-1000)
            xmax_norm: Normalized right x (0-1000)
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            BoundingBox with pixel coordinates
        """
        xtl = (xmin_norm / 1000.0) * image_width
        ytl = (ymin_norm / 1000.0) * image_height
        xbr = (xmax_norm / 1000.0) * image_width
        ybr = (ymax_norm / 1000.0) * image_height
        return cls(xtl, ytl, xbr, ybr)

    def __repr__(self):
        return f"BoundingBox(xtl={self.xtl}, ytl={self.ytl}, xbr={self.xbr}, ybr={self.ybr})"


class BBoxAnnotator:
    """
    Interface for bounding box annotation.

    This class provides methods for detecting and annotating objects
    with bounding boxes using natural language descriptions.
    """

    def __init__(self,client: gemini_client.GeminiClient):
        """Initialize bounding box annotator."""
        logger.info("Initialized BBoxAnnotator")
        self.client = client

    def annotate_single_object(
        self,
        image: Union[Path, Image.Image],
        description: str
    ) -> BoundingBox:
        """
        Detect and annotate a single object in an image.

        Args:
            image: Image file path or PIL Image object
            description: Natural language description of the object
                        (e.g., "the player wearing red jersey number 10")

        Returns:
            BoundingBox for the detected object

        Raises:
            NotImplementedError: This method needs to be implemented

        Example:
            >>> annotator = BBoxAnnotator()
            >>> bbox = annotator.annotate_single_object(
            ...     image_path,
            ...     "the scoreboard in the upper left corner"
            ... )
            >>> print(bbox.to_list())
            [100, 50, 300, 150]
        """
       # Prepare image bytes and mime type
        if isinstance(image, Path):
            img = Image.open(image)
            image_bytes = image.read_bytes()
            mime_type = f"image/{img.format.lower() if img.format else 'jpeg'}"
        elif isinstance(image, Image.Image):
            img = image
            import io
            buf = io.BytesIO()
            format = img.format if img.format else "JPEG"
            img.save(buf, format=format)
            image_bytes = buf.getvalue()
            mime_type = f"image/{img.format.lower() if img.format else 'jpeg'}"
        else:
            raise ValueError("image must be a Path or PIL.Image.Image object")

        width, height = img.size

        # call grounding model to get normalized bbox
        bbox_norm = self.client.ground_bounding_box(
            image_bytes=image_bytes,
            mime_type=mime_type,
            description=description,
            task_type="single_box"
        )

        # convert normalized bbox to pixel bbox
        ytl_norm, xtl_norm, ybr_norm, xbr_norm = bbox_norm
        bbox = BoundingBox.from_normalized(
            ymin_norm=ytl_norm,
            xmin_norm=xtl_norm,
            ymax_norm=ybr_norm,
            xmax_norm=xbr_norm,
            image_width=width,
            image_height=height
        )

        return bbox
    


    def annotate_multiple_objects(
        self,
        image: Union[Path, Image.Image],
        descriptions: List[str]
    ) -> List[BoundingBox]:
        """
        Detect and annotate multiple objects in an image.

        Args:
            image: Image file path or PIL Image object
            descriptions: List of natural language descriptions for each object

        Returns:
            List of BoundingBox objects, one for each description

        Raises:
            NotImplementedError: This method needs to be implemented

        Example:
            >>> annotator = BBoxAnnotator()
            >>> bboxes = annotator.annotate_multiple_objects(
            ...     image_path,
            ...     [
            ...         "the player wearing red jersey number 10",
            ...         "the player wearing blue jersey number 5"
            ...     ]
            ... )
            >>> for bbox in bboxes:
            ...     print(bbox.to_list())
            [100, 200, 200, 400]
            [300, 250, 400, 450]
        """
        # Prepare image bytes and mime type
        if isinstance(image, Path):
            img = Image.open(image)
            image_bytes = image.read_bytes()
            mime_type = f"image/{img.format.lower() if img.format else 'jpeg'}"
        elif isinstance(image, Image.Image):
            img = image
            import io
            buf = io.BytesIO()
            format = img.format if img.format else "JPEG"
            img.save(buf, format=format)
            image_bytes = buf.getvalue()
            mime_type = f"image/{img.format.lower() if img.format else 'jpeg'}"
        else:
            raise ValueError("image must be a Path or PIL.Image.Image object")

        width, height = img.size

        
        discription = "\n".join(descriptions)
        normalized_bboxes_list = self.client.ground_bounding_box(
            image_bytes=image_bytes,
            mime_type=mime_type,
            description=discription,
            task_type="multiple_boxes"
        )
        # normalized_bboxes_list is expected to be a List[Tuple[float, float, float, float]]

        # Convert normalized bboxes to pixel bboxes
        bboxes: List[BoundingBox] = []
        for bbox_norm in normalized_bboxes_list:
            ytl_norm, xtl_norm, ybr_norm, xbr_norm = bbox_norm
            bbox = BoundingBox.from_normalized(
                ymin_norm=ytl_norm,
                xmin_norm=xtl_norm,
                ymax_norm=ybr_norm,
                xmax_norm=xbr_norm,
                image_width=width,
                image_height=height
            )
            bboxes.append(bbox)

        return bboxes
    

    def extract_frame_from_video(
        self,
        video_path: Path,
        frame_number: int,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Extract a specific frame from a video.

        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (0-indexed)
            output_path: Optional output path for the frame image
                        If None, will create a temp file

        Returns:
            Path to the extracted frame image

        Raises:
            NotImplementedError: This method needs to be implemented

        Note:
            This is a utility method that might be useful for frame-based
            annotation tasks. You can implement this using OpenCV or similar.
        """
        raise NotImplementedError(
            "extract_frame_from_video() needs to be implemented. "
            "This should extract a frame from video and save it as an image."
        )


# Example usage documentation
"""
Example usage for your colleague to implement:

```python
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator, BoundingBox
from pathlib import Path

# Initialize annotator
annotator = BBoxAnnotator()

# Single object detection
image_path = Path("frame_001.jpg")
description = "the scoreboard in the upper left corner showing team scores"
bbox = annotator.annotate_single_object(image_path, description)
print(f"Detected bbox: {bbox.to_list()}")

# Multiple objects detection
descriptions = [
    "the player wearing red jersey number 10",
    "the player wearing blue jersey number 5"
]
bboxes = annotator.annotate_multiple_objects(image_path, descriptions)
for i, bbox in enumerate(bboxes):
    print(f"Object {i}: {bbox.to_list()}")

# Extract frame from video
video_path = Path("game.mp4")
frame_path = annotator.extract_frame_from_video(video_path, frame_number=100)
```

Implementation notes:
1. You can use Gemini's grounding capabilities or other models like Grounding DINO
2. Make sure to handle cases where objects are not found
3. Consider adding confidence scores to the BoundingBox class
4. Add proper error handling and logging
"""
