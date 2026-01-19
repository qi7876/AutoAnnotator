"""Configuration management module."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class GeminiConfig(BaseModel):
    """Gemini API configuration."""

    model_backend: str = "ai_studio"
    grounding_backend: str = "ai_studio"
    model_api_key: str = ""
    grounding_api_key: str = ""
    gcs_bucket: str = ""
    gcs_prefix: str = ""
    gcs_sync_delete: bool = True
    model: str = "gemini-2.5-flash"
    grounding_model: str = "gemini-robotics-er-1.5-preview"
    model_thinking_level: str = "high"
    grounding_thinking_budget: int = 0
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    video: Dict[str, Any] = Field(default_factory=dict)
    video_sampling_fps: int = 1


class OutputConfig(BaseModel):
    """Output directory configuration."""

    temp_dir: str = "data/output/temp"


class BatchProcessingConfig(BaseModel):
    num_workers: int = 1
    enable_clips: bool = True
    enable_frames: bool = True


class TasksConfig(BaseModel):
    """Tasks configuration."""

    tracking: Dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/annotator.log"


class Config(BaseModel):
    """Main configuration class."""

    gemini: GeminiConfig
    output: OutputConfig
    batch_processing: BatchProcessingConfig = Field(
        default_factory=BatchProcessingConfig
    )
    tasks: TasksConfig
    logging: LoggingConfig

    # Runtime settings
    api_key: str = Field(default="")
    project_root: Path = Field(default_factory=Path.cwd)
    dataset_root: Path = Field(default_factory=Path.cwd)

    @field_validator("project_root", "dataset_root", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class ConfigManager:
    """Manages application configuration."""

    _instance = None
    _config: Optional[Config] = None

    def __new__(cls):
        """Singleton pattern to ensure only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize config manager."""
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load configuration from files."""
        # Load environment variables
        env_file = Path(__file__).parent.parent.parent / "config" / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        # Load YAML config
        config_file = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Add environment variables
        config_data["api_key"] = ""
        config_data.setdefault("gemini", {})
        model_api_key = os.getenv("GEMINI_MODEL_API_KEY", "").strip()
        if model_api_key:
            config_data["gemini"]["model_api_key"] = model_api_key
        grounding_api_key = os.getenv("GEMINI_GROUNDING_API_KEY", "").strip()
        if grounding_api_key:
            config_data["gemini"]["grounding_api_key"] = grounding_api_key
        config_data["project_root"] = os.getenv(
            "PROJECT_ROOT", str(Path(__file__).parent.parent.parent)
        )
        config_data["dataset_root"] = os.getenv(
            "DATASET_ROOT", str(Path.cwd() / "data" / "Dataset")
        )

        # Validate API key
        if not config_data["gemini"].get("model_api_key"):
            raise ValueError(
                "GEMINI_MODEL_API_KEY not found in environment variables. "
                "Please copy config/.env.example to config/.env and add your API key."
            )
        if not config_data["gemini"].get("grounding_api_key"):
            raise ValueError(
                "GEMINI_GROUNDING_API_KEY not found in environment variables. "
                "Please set GEMINI_GROUNDING_API_KEY."
            )
        if not config_data["gemini"].get("model_backend"):
            raise ValueError(
                "gemini.model_backend not configured. Please set it in config/config.yaml."
            )
        if not config_data["gemini"].get("grounding_backend"):
            raise ValueError(
                "gemini.grounding_backend not configured. Please set it in config/config.yaml."
            )

        # Create config object
        self._config = Config(**config_data)

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary output directories."""
        temp_dir = Path(self._config.project_root) / self._config.output.temp_dir
        log_dir = (
            Path(self._config.project_root) / Path(self._config.logging.file).parent
        )

        temp_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> Config:
        """Get configuration object."""
        assert self._config is not None
        return self._config

    def get_prompt_path(self, task_name: str) -> Path:
        """Get path to prompt template file."""
        prompts_dir = Path(self._config.project_root) / "config" / "prompts"
        return prompts_dir / f"{task_name.lower()}.md"

    def get_temp_output_path(self, clip_id: str) -> Path:
        """Get temporary output path for a clip."""
        temp_dir = Path(self._config.project_root) / self._config.output.temp_dir
        return temp_dir / f"{clip_id}.json"


# Global config instance
def get_config() -> Config:
    """Get global config instance."""
    return ConfigManager().config


def get_config_manager() -> ConfigManager:
    """Get global config manager instance."""
    return ConfigManager()
