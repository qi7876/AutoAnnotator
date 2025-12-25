"""Configuration management module."""

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class GeminiConfig(BaseModel):
    """Gemini API configuration."""

    model: str = "gemini-2.5-flash"
    grounding_model: str = "gemini-robotics-er-1.5-preview"
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    video: Dict[str, Any] = Field(default_factory=dict)
    video_sampling_fps: int = 1


class OutputConfig(BaseModel):
    """Output directory configuration."""

    temp_dir: str = "data/output/temp"
    final_dir: str = "data/output/final"
    keep_temp_files: bool = False


class TasksConfig(BaseModel):
    """Tasks configuration."""

    enabled: List[str] = Field(default_factory=list)
    scoreboard_single: Dict[str, Any] = Field(default_factory=dict)
    scoreboard_multiple: Dict[str, Any] = Field(default_factory=dict)
    spatial_relationships: Dict[str, Any] = Field(default_factory=dict)
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
    _config: Config = None

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
        config_data["api_key"] = os.getenv("GEMINI_API_KEY", "")
        config_data["project_root"] = os.getenv(
            "PROJECT_ROOT",
            str(Path(__file__).parent.parent.parent)
        )
        config_data["dataset_root"] = os.getenv(
            "DATASET_ROOT",
            str(Path.cwd() / "data" / "Dataset")
        )

        # Validate API key
        if not config_data["api_key"]:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please copy config/.env.example to config/.env and add your API key."
            )

        # Create config object
        self._config = Config(**config_data)

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary output directories."""
        temp_dir = Path(self._config.project_root) / self._config.output.temp_dir
        final_dir = Path(self._config.project_root) / self._config.output.final_dir
        log_dir = Path(self._config.project_root) / Path(self._config.logging.file).parent

        temp_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> Config:
        """Get configuration object."""
        return self._config

    def get_prompt_path(self, task_name: str) -> Path:
        """Get path to prompt template file."""
        prompts_dir = Path(self._config.project_root) / "config" / "prompts"
        return prompts_dir / f"{task_name.lower()}.md"

    def get_temp_output_path(self, clip_id: str) -> Path:
        """Get temporary output path for a clip."""
        temp_dir = Path(self._config.project_root) / self._config.output.temp_dir
        return temp_dir / f"{clip_id}.json"

    def is_task_enabled(self, task_name: str) -> bool:
        """Check if a task is enabled."""
        return task_name in self._config.tasks.enabled


# Global config instance
def get_config() -> Config:
    """Get global config instance."""
    return ConfigManager().config


def get_config_manager() -> ConfigManager:
    """Get global config manager instance."""
    return ConfigManager()
