"""Prompt template loader."""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loader for prompt templates."""

    # Map task names to prompt file names
    TASK_TO_PROMPT_FILE = {
        "ScoreboardSingle": "scoreboardsingle.md",
        "ScoreboardMultiple": "scoreboardmultiple.md",
        "Objects_Spatial_Relationships": "objects_spatial_relationships.md",
        "Spatial_Temporal_Grounding": "spatial_temporal_grounding.md",
        "Continuous_Actions_Caption": "continuous_actions_caption.md",
        "Continuous_Events_Caption": "continuous_events_caption.md",
        "Object_Tracking": "object_tracking.md",
    }

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt loader.

        Args:
            prompts_dir: Directory containing prompt templates
                        If None, uses default config/prompts directory
        """
        if prompts_dir is None:
            # Default to config/prompts relative to project root
            self.prompts_dir = (
                Path(__file__).parent.parent.parent.parent / "config" / "prompts"
            )
        else:
            self.prompts_dir = prompts_dir

        if not self.prompts_dir.exists():
            raise FileNotFoundError(
                f"Prompts directory not found: {self.prompts_dir}"
            )

        logger.info(f"Initialized PromptLoader with dir: {self.prompts_dir}")

    def load_prompt(self, task_name: str, **kwargs) -> str:
        """
        Load and format prompt template for a task.

        Args:
            task_name: Task name (e.g., "ScoreboardSingle")
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If task name is not recognized
            FileNotFoundError: If prompt file doesn't exist
        """
        if task_name not in self.TASK_TO_PROMPT_FILE:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Valid tasks: {list(self.TASK_TO_PROMPT_FILE.keys())}"
            )

        prompt_file = self.prompts_dir / self.TASK_TO_PROMPT_FILE[task_name]

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        # Load template
        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()

        # Format template with provided variables
        try:
            formatted = template.format(**kwargs)
            logger.debug(f"Loaded and formatted prompt for {task_name}")
            return formatted
        except KeyError as e:
            raise ValueError(
                f"Missing required variable in prompt template: {e}"
            )

    def get_required_variables(self, task_name: str) -> list[str]:
        """
        Get list of required variables for a task prompt.

        Args:
            task_name: Task name

        Returns:
            List of variable names

        Note:
            This parses the template for {variable} placeholders
        """
        if task_name not in self.TASK_TO_PROMPT_FILE:
            raise ValueError(f"Unknown task: {task_name}")

        prompt_file = self.prompts_dir / self.TASK_TO_PROMPT_FILE[task_name]

        with open(prompt_file, "r", encoding="utf-8") as f:
            template = f.read()

        # Extract variable names from {variable} patterns
        import re
        variables = re.findall(r'\{(\w+)\}', template)

        return list(set(variables))

    def list_available_tasks(self) -> list[str]:
        """
        Get list of all available task names.

        Returns:
            List of task names
        """
        return list(self.TASK_TO_PROMPT_FILE.keys())

    def validate_prompt_files(self) -> Dict[str, bool]:
        """
        Validate that all prompt files exist.

        Returns:
            Dictionary mapping task names to existence status
        """
        status = {}
        for task_name, filename in self.TASK_TO_PROMPT_FILE.items():
            prompt_file = self.prompts_dir / filename
            status[task_name] = prompt_file.exists()

        return status
