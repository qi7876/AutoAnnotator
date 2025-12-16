"""AutoAnnotator - AI-powered video annotation system."""

__version__ = "0.1.0"

from .adapters import InputAdapter, ClipMetadata
from .annotators import GeminiClient, TaskAnnotatorFactory
from .config import get_config, get_config_manager
from .utils import JSONUtils, PromptLoader, VideoUtils

__all__ = [
    "InputAdapter",
    "ClipMetadata",
    "GeminiClient",
    "TaskAnnotatorFactory",
    "get_config",
    "get_config_manager",
    "JSONUtils",
    "PromptLoader",
    "VideoUtils",
]
