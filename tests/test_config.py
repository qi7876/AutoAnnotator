"""Tests for configuration management."""

import pytest
from pathlib import Path

from auto_annotator.config import ConfigManager, get_config


def test_config_manager_singleton():
    """Test that ConfigManager is a singleton."""
    manager1 = ConfigManager()
    manager2 = ConfigManager()
    assert manager1 is manager2


def test_get_config():
    """Test getting configuration."""
    config = get_config()
    assert config is not None
    assert hasattr(config, "gemini")
    assert hasattr(config, "output")
    assert hasattr(config, "tasks")
    assert hasattr(config, "logging")


def test_config_has_api_key():
    """Test that API key is loaded."""
    config = get_config()
    # Note: This will fail if .env is not set up
    # In actual testing, you might want to mock this
    assert config.api_key != ""


def test_prompt_path():
    """Test getting prompt template paths."""
    manager = ConfigManager()
    path = manager.get_prompt_path("ScoreboardSingle")
    assert path.name == "scoreboardsingle.md"


def test_temp_output_path():
    """Test getting temp output paths."""
    manager = ConfigManager()
    path = manager.get_temp_output_path("test_segment_001")
    assert "test_segment_001.json" in str(path)


def test_is_task_enabled():
    """Test checking if tasks are enabled."""
    manager = ConfigManager()
    config = manager.config

    for task in config.tasks.enabled:
        assert manager.is_task_enabled(task)

    assert not manager.is_task_enabled("NonExistentTask")
