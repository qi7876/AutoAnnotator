"""Tests for configuration management."""

from pathlib import Path

from auto_annotator.config import ConfigManager, get_config


def _reset_config_manager_singleton() -> None:
    ConfigManager._instance = None
    ConfigManager._config = None


def test_config_manager_singleton(monkeypatch):
    """Test that ConfigManager is a singleton."""
    _reset_config_manager_singleton()
    monkeypatch.setenv("GEMINI_MODEL_API_KEY", "test-model-key")
    monkeypatch.setenv("GEMINI_GROUNDING_API_KEY", "test-grounding-key")
    manager1 = ConfigManager()
    manager2 = ConfigManager()
    assert manager1 is manager2


def test_get_config(monkeypatch):
    """Test getting configuration."""
    _reset_config_manager_singleton()
    monkeypatch.setenv("GEMINI_MODEL_API_KEY", "test-model-key")
    monkeypatch.setenv("GEMINI_GROUNDING_API_KEY", "test-grounding-key")
    config = get_config()
    assert config is not None
    assert hasattr(config, "gemini")
    assert hasattr(config, "output")
    assert hasattr(config, "tasks")
    assert hasattr(config, "logging")


def test_config_has_api_key(monkeypatch):
    """Test that API keys are loaded into gemini config."""
    _reset_config_manager_singleton()
    monkeypatch.setenv("GEMINI_MODEL_API_KEY", "test-model-key")
    monkeypatch.setenv("GEMINI_GROUNDING_API_KEY", "test-grounding-key")
    config = get_config()
    assert config.gemini.model_api_key != ""
    assert config.gemini.grounding_api_key != ""


def test_prompt_path(monkeypatch):
    """Test getting prompt template paths."""
    _reset_config_manager_singleton()
    monkeypatch.setenv("GEMINI_MODEL_API_KEY", "test-model-key")
    monkeypatch.setenv("GEMINI_GROUNDING_API_KEY", "test-grounding-key")
    manager = ConfigManager()
    path = manager.get_prompt_path("ScoreboardSingle")
    assert path.name == "scoreboardsingle.md"


def test_temp_output_path(monkeypatch):
    """Test getting temp output paths."""
    _reset_config_manager_singleton()
    monkeypatch.setenv("GEMINI_MODEL_API_KEY", "test-model-key")
    monkeypatch.setenv("GEMINI_GROUNDING_API_KEY", "test-grounding-key")
    manager = ConfigManager()
    path = manager.get_temp_output_path("test_segment_001")
    assert "test_segment_001.json" in str(path)


def test_tasks_config_present(monkeypatch):
    """Test task config section exists and loads."""
    _reset_config_manager_singleton()
    monkeypatch.setenv("GEMINI_MODEL_API_KEY", "test-model-key")
    monkeypatch.setenv("GEMINI_GROUNDING_API_KEY", "test-grounding-key")
    manager = ConfigManager()
    assert isinstance(manager.config.tasks.tracking, dict)
