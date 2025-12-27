"""Gemini API client for video annotation."""

import json
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types

from ..config import get_config

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Gemini API."""

    def __init__(self):
        """Initialize Gemini client."""
        self.config = get_config()

        self.client = genai.Client(api_key=self.config.api_key)
        self.model_name = self.config.gemini.model
        self.grounding_model_name = self.config.gemini.grounding_model

        self.generation_config = self.config.gemini.generation_config

        logger.info(
            "Initialized Gemini client with model=%s grounding_model=%s",
            self.model_name,
            self.grounding_model_name
        )

    def _build_generation_config(
        self,
        overrides: Optional[Dict[str, Any]] = None
    ) -> types.GenerateContentConfig:
        """Build a typed GenerateContentConfig from config dict."""
        if isinstance(self.generation_config, types.GenerateContentConfig):
            base_config = self.generation_config
        else:
            base_config = types.GenerateContentConfig(**self.generation_config)

        if not overrides:
            return base_config

        merged = base_config.model_dump()
        merged.update(overrides)
        return types.GenerateContentConfig(**merged)

    def upload_video(self, video_path: Path) -> Any:
        """
        Upload a video file to Gemini File API.

        Args:
            video_path: Path to the video file

        Returns:
            Uploaded file object

        Raises:
            FileNotFoundError: If video file doesn't exist
            Exception: If upload fails
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Uploading video: {video_path}")

        try:
            video_file = self.client.files.upload(file=video_path)
            logger.info(f"Uploaded file '{video_file.name}' as: {video_file.uri}")

            # Wait for file to be processed
            timeout = self.config.gemini.video["upload_timeout_sec"]
            start_time = time.time()

            while video_file.state.name == "PROCESSING":
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Video processing timeout after {timeout} seconds"
                    )

                logger.info("Waiting for video processing...")
                time.sleep(2)
                video_file = self.client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_file.state}")

            logger.info("Video processing complete")
            return video_file

        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            raise

    def annotate_video(
        self,
        video_file: Any,
        prompt: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Annotate a video using Gemini API.

        Args:
            video_file: Uploaded video file object
            prompt: Annotation prompt
            timeout: Optional timeout in seconds

        Returns:
            Annotation result as dictionary

        Raises:
            Exception: If annotation fails
        """
        if timeout is None:
            timeout = self.config.gemini.video["processing_timeout_sec"]

        request_config = self._build_generation_config()
        logger.info("Generating annotation with model=%s", self.model_name)

        try:
            video_file_uri = video_file.uri
            video_part = types.Part(
                file_data=types.FileData(
                    file_uri=video_file_uri,
                    mime_type="video/mp4"
                ),
                video_metadata=types.VideoMetadata(
                    fps=self.config.gemini.video_sampling_fps
                )
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[video_part, prompt],
                config=request_config
            )

            result = self._parse_json_response(response.text)

            logger.info("Annotation generated successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to generate annotation: {e}")
            raise

    def annotate_image(
        self,
        image: Union[Path, bytes],
        prompt: str,
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Annotate an image using Gemini API.

        Args:
            image: Image file path or raw bytes
            prompt: Annotation prompt
            mime_type: Optional MIME type (required when passing bytes)

        Returns:
            Annotation result as dictionary
        """
        request_config = self._build_generation_config()
        logger.info("Generating image annotation with model=%s", self.model_name)

        try:
            if isinstance(image, Path):
                image_bytes = image.read_bytes()
                if mime_type is None:
                    mime_type = mimetypes.guess_type(str(image))[0]
            else:
                image_bytes = image
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type or "image/jpeg"
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[image_part, prompt],
                config=request_config,
            )

            result = self._parse_json_response(response.text)

            logger.info("Image annotation generated successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to generate image annotation: {e}")
            raise

    def ground_bounding_box(
        self,
        image_bytes: bytes,
        mime_type: str,
        description: str,
        task_type: str
    ) -> List[float]:
        """
        Generate bounding box coordinates using grounding model.

        Args:
            image_bytes: Image content as bytes.
            mime_type: The MIME type of the image (e.g., "image/jpeg").
            description: Natural language description of the object.

        Returns:
            Bounding box coordinates [xtl, ytl, xbr, ybr] normalized to [0, 1000].
        
        Raises:
            RuntimeError: If model response is invalid or parsing fails.
        """
        logger.info(f"Generating bounding box for: {description}")
        MODEL_ID = self.grounding_model_name
        client = self.client

        if task_type == "single_box":   
            prompt = f"""
                Detect a single object that matches this description: "{description}".
                Return a JSON array with exactly one element, containing only the bounding box coordinates:
                [{{"box_2d": [ymin, xmin, ymax, xmax]}}]
                Coordinates are normalized in the range 0-1000. Do not return masks, labels, or extra objects.
                """
        elif task_type == "multiple_boxes":
            prompt = f"""
                Detect all objects that match this description: "{description}".
                Return a JSON array with one or more elements, each containing the bounding box coordinates:
                [{{"box_2d": [ymin, xmin, ymax, xmax]}}]
                Coordinates are normalized in the range 0-1000. Do not return masks, labels, or extra objects.
                """
        else:
            raise ValueError(f"Invalid task type: {task_type}")

        try:
            # Call grounding model using the core client
            image_response = client.models.generate_content(
                model=MODEL_ID,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
                config=types.GenerateContentConfig(temperature=0, thinking_config=types.ThinkingConfig(thinking_budget=0))
            )

            data = self._parse_json_response(image_response.text)

            # extract bounding box
            if task_type == "single_box":
                if not data or not isinstance(data, list) or len(data) != 1:
                    raise RuntimeError(f"Unexpected response format: {image_response.text}")
                bbox_norm = data[0]["box_2d"]
                return bbox_norm
            elif task_type == "multiple_boxes":
                if not data or not isinstance(data, list):
                    raise RuntimeError(f"Unexpected response format: {image_response.text}")
                bbox_norms = [box["box_2d"] for box in data if "box_2d" in box]
                return bbox_norms

        except Exception as e:
            logger.error(f"Failed to generate bounding box: {e}")
            raise

    def _parse_json_response(self, response_text: str) -> Any:
        """
        Parse JSON response from Gemini.

        Args:
            response_text: Raw response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If response is not valid JSON
        """
        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {text}")
            raise ValueError(f"Invalid JSON response: {e}")

    def cleanup_file(self, file_obj: Any):
        """
        Delete uploaded file from Gemini File API.

        Args:
            file_obj: Uploaded file object
        """
        try:
            self.client.files.delete(name=file_obj.name)
            logger.info(f"Deleted file: {file_obj.name}")
        except Exception as e:
            logger.warning(f"Failed to delete file: {e}")
