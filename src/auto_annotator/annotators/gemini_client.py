"""Gemini API client for video annotation."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
from google.generativeai import upload_file, get_file
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..config import get_config

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Gemini API."""

    def __init__(self):
        """Initialize Gemini client."""
        self.config = get_config()

        # Configure API
        genai.configure(api_key=self.config.api_key)

        # Initialize models
        self.model = genai.GenerativeModel(
            model_name=self.config.gemini.model,
            generation_config=self.config.gemini.generation_config,
        )

        # Grounding model for bounding box detection
        self.grounding_model = genai.GenerativeModel(
            model_name=self.config.gemini.grounding_model
        )

        # Safety settings - disable blocking for annotation tasks
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        logger.info(f"Initialized Gemini client with model: {self.config.gemini.model}")

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
            # Upload file
            video_file = upload_file(
                path=str(video_path),
                display_name=video_path.name
            )

            logger.info(f"Uploaded file '{video_file.display_name}' as: {video_file.uri}")

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
                video_file = get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_file.state}")

            logger.info("Video processing complete")
            return video_file

        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            raise

    def upload_image(self, image_path: Path) -> Any:
        """
        Upload an image file to Gemini File API.

        Args:
            image_path: Path to the image file

        Returns:
            Uploaded file object
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"Uploading image: {image_path}")

        try:
            image_file = upload_file(
                path=str(image_path),
                display_name=image_path.name
            )
            logger.info(f"Uploaded image as: {image_file.uri}")
            return image_file

        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
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

        logger.info("Generating annotation...")

        try:
            # Generate content
            response = self.model.generate_content(
                [video_file, prompt],
                safety_settings=self.safety_settings,
                request_options={"timeout": timeout}
            )

            # Parse JSON response
            result = self._parse_json_response(response.text)

            logger.info("Annotation generated successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to generate annotation: {e}")
            raise

    def annotate_image(
        self,
        image_file: Any,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Annotate an image using Gemini API.

        Args:
            image_file: Uploaded image file object
            prompt: Annotation prompt

        Returns:
            Annotation result as dictionary
        """
        logger.info("Generating image annotation...")

        try:
            response = self.model.generate_content(
                [image_file, prompt],
                safety_settings=self.safety_settings,
            )

            result = self._parse_json_response(response.text)

            logger.info("Image annotation generated successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to generate image annotation: {e}")
            raise

    def ground_bounding_box(
        self,
        image_file: Any,
        description: str
    ) -> List[float]:
        """
        Generate bounding box coordinates using grounding model.

        Args:
            image_file: Uploaded image file object
            description: Natural language description of the object

        Returns:
            Bounding box coordinates [xtl, ytl, xbr, ybr] normalized to [0, 1000]

        Note:
            This is a placeholder interface. The actual implementation
            will be completed by your colleague working on bounding box annotation.
        """
        logger.info(f"Generating bounding box for: {description}")

        # TODO: Implement actual grounding logic
        # This should call the Gemini grounding model and return coordinates
        raise NotImplementedError(
            "Bounding box grounding not yet implemented. "
            "To be completed by colleague working on bounding box annotation."
        )

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
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
            file_obj.delete()
            logger.info(f"Deleted file: {file_obj.name}")
        except Exception as e:
            logger.warning(f"Failed to delete file: {e}")
