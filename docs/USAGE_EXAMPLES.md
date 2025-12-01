# Usage Examples

This document provides detailed examples of how to use AutoAnnotator in various scenarios.

## Basic Usage

### Example 1: Annotating a Single Segment

```python
from pathlib import Path
from auto_annotator.main import process_segment
from auto_annotator import (
    InputAdapter,
    GeminiClient,
    PromptLoader
)
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.config import get_config

# Setup
config = get_config()
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator()
tracker = ObjectTracker()
output_dir = Path("output/temp")

# Load segment
segment_metadata = InputAdapter.load_from_json(
    Path("segments/basketball_segment_001.json")
)

# Process
output_path = process_segment(
    segment_metadata=segment_metadata,
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker,
    output_dir=output_dir
)

print(f"Annotations saved to: {output_path}")
```

### Example 2: Batch Processing Multiple Segments

```python
from pathlib import Path
from auto_annotator.main import process_segments_batch

# Process all segments in a directory
segments_dir = Path("segments/3x3_Basketball/Men")
output_dir = Path("output/temp")

process_segments_batch(
    segment_paths=list(segments_dir.glob("*.json")),
    output_dir=output_dir
)
```

### Example 3: Processing Specific Tasks Only

```python
from auto_annotator import InputAdapter

# Load segment metadata
segment_metadata = InputAdapter.load_from_json(
    Path("segments/segment_001.json")
)

# Filter to only specific tasks
segment_metadata.tasks_to_annotate = [
    "ScoreboardSingle",
    "Continuous_Actions_Caption"
]

# Then process as normal...
```

## Working with Individual Tasks

### Example 4: Scoreboard Understanding (Single Frame)

```python
from auto_annotator import (
    TaskAnnotatorFactory,
    GeminiClient,
    PromptLoader
)
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker

# Initialize
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator()
tracker = ObjectTracker()

# Create annotator
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="ScoreboardSingle",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

# Annotate
result = annotator.annotate(segment_metadata)

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Bounding box: {result['bounding_box']}")
```

### Example 5: Continuous Actions Caption

```python
# Create annotator for continuous actions
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="Continuous_Actions_Caption",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

# Annotate
result = annotator.annotate(segment_metadata)

# Process results
for i, (window, action) in enumerate(
    zip(result['A_window_frame'], result['answer'])
):
    print(f"Action {i+1}: {action}")
    print(f"  Time window: frames {window}")
```

### Example 6: Object Tracking

```python
# Create tracking annotator
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="Object_Tracking",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

# Annotate
result = annotator.annotate(segment_metadata)

print(f"Tracking query: {result['query']}")
print(f"Tracking window: {result['Q_window_frame']}")
print(f"First bbox: {result.get('first_bounding_box', 'Not implemented')}")
```

## Utility Functions

### Example 7: Video Information Extraction

```python
from pathlib import Path
from auto_annotator.utils import VideoUtils

# Get video metadata
video_path = Path("Dataset/3x3_Basketball/Men/1.mp4")
info = VideoUtils.get_video_info(video_path)

print(f"FPS: {info['fps']}")
print(f"Total frames: {info['total_frames']}")
print(f"Resolution: {info['resolution']}")
print(f"Duration: {info['duration_sec']} seconds")

# Extract a specific frame
frame_path = VideoUtils.extract_frame(
    video_path=video_path,
    frame_number=100,
    output_path=Path("output/frame_100.jpg")
)

# Convert between frames and seconds
frames = VideoUtils.seconds_to_frames(5.0, fps=10)
print(f"5 seconds = {frames} frames at 10 FPS")

seconds = VideoUtils.frames_to_seconds(50, fps=10)
print(f"50 frames = {seconds} seconds at 10 FPS")
```

### Example 8: JSON Manipulation

```python
from pathlib import Path
from auto_annotator.utils import JSONUtils

# Load existing annotation file
base_json = JSONUtils.load_json(Path("Dataset/3x3_Basketball/Men/1.json"))

# Load new annotations from temp directory
new_annotations = []
for temp_file in Path("output/temp").glob("*.json"):
    data = JSONUtils.load_json(temp_file)
    new_annotations.extend(data["annotations"])

# Merge annotations
merged = JSONUtils.merge_annotations(base_json, new_annotations)

# Save merged result
JSONUtils.save_json(merged, Path("output/final/1.json"))

# Validate result
is_valid, error = JSONUtils.validate_annotation_json(merged)
if is_valid:
    print("Validation successful!")
else:
    print(f"Validation failed: {error}")
```

### Example 9: Filtering Annotations

```python
from auto_annotator.utils import JSONUtils

# Load annotation file
data = JSONUtils.load_json(Path("output/final/1.json"))

# Filter by task type
understanding_tasks = JSONUtils.filter_annotations_by_task(
    data, task_l1="Understanding"
)
print(f"Found {len(understanding_tasks)} Understanding tasks")

perception_tasks = JSONUtils.filter_annotations_by_task(
    data, task_l1="Perception"
)
print(f"Found {len(perception_tasks)} Perception tasks")

# Filter by specific task
scoreboard_annotations = JSONUtils.filter_annotations_by_task(
    data, task_l2="ScoreboardSingle"
)
print(f"Found {len(scoreboard_annotations)} Scoreboard (Single) annotations")

# Get all annotation IDs
all_ids = JSONUtils.get_annotation_ids(data)
print(f"Total annotations: {len(all_ids)}")
```

## Custom Prompt Templates

### Example 10: Creating Custom Prompt Variables

```python
from auto_annotator.utils import PromptLoader

# Initialize prompt loader
loader = PromptLoader()

# Load prompt with custom variables
prompt = loader.load_prompt(
    task_name="Continuous_Actions_Caption",
    num_first_frame=150,
    total_frames=100,
    fps=10,
    duration_sec=10.0
)

print(prompt)
```

### Example 11: Checking Required Variables

```python
from auto_annotator.utils import PromptLoader

loader = PromptLoader()

# Get required variables for a task
variables = loader.get_required_variables("ScoreboardMultiple")
print(f"Required variables: {variables}")

# Validate all prompt files
status = loader.validate_prompt_files()
for task, exists in status.items():
    if not exists:
        print(f"WARNING: Missing prompt file for {task}")
```

## Configuration Management

### Example 12: Accessing Configuration

```python
from auto_annotator import get_config, get_config_manager

# Get config object
config = get_config()

# Access settings
print(f"Gemini model: {config.gemini.model}")
print(f"Output directory: {config.output.temp_dir}")
print(f"Enabled tasks: {config.tasks.enabled}")

# Get config manager for advanced operations
manager = get_config_manager()

# Check if task is enabled
if manager.is_task_enabled("ScoreboardSingle"):
    print("ScoreboardSingle is enabled")

# Get prompt path
prompt_path = manager.get_prompt_path("Object_Tracking")
print(f"Prompt template: {prompt_path}")
```

## Error Handling

### Example 13: Robust Processing with Error Handling

```python
import logging
from pathlib import Path
from auto_annotator import InputAdapter
from auto_annotator.main import process_segment

logger = logging.getLogger(__name__)

segments_dir = Path("segments")
successful = []
failed = []

for segment_file in segments_dir.glob("*.json"):
    try:
        # Load and validate
        metadata = InputAdapter.load_from_json(segment_file)
        is_valid, error = InputAdapter.validate_metadata(metadata)

        if not is_valid:
            logger.error(f"Invalid metadata: {error}")
            failed.append((segment_file, error))
            continue

        # Process
        output_path = process_segment(...)
        successful.append((segment_file, output_path))

    except Exception as e:
        logger.error(f"Failed to process {segment_file}: {e}")
        failed.append((segment_file, str(e)))

# Report
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

for segment_file, error in failed:
    print(f"  - {segment_file.name}: {error}")
```

## Advanced Usage

### Example 14: Custom Annotator Implementation

```python
from auto_annotator.annotators.base_annotator import BaseAnnotator

class CustomAnnotator(BaseAnnotator):
    """Custom annotator for special tasks."""

    def get_task_name(self) -> str:
        return "CustomTask"

    def get_task_l1(self) -> str:
        return "Understanding"

    def annotate(self, segment_metadata):
        # Custom annotation logic
        prompt = self.load_prompt(segment_metadata)

        # Upload and process
        video_file = self.gemini_client.upload_video(
            segment_metadata.get_video_path()
        )

        result = self.gemini_client.annotate_video(video_file, prompt)

        # Add metadata
        result = self.add_metadata_fields(result)

        # Cleanup
        self.gemini_client.cleanup_file(video_file)

        return result
```

### Example 15: Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from auto_annotator.main import process_segment

def process_segment_wrapper(segment_path):
    """Wrapper for parallel processing."""
    try:
        metadata = InputAdapter.load_from_json(segment_path)
        output_path = process_segment(...)
        return (True, segment_path, output_path)
    except Exception as e:
        return (False, segment_path, str(e))

# Process segments in parallel
segments = list(Path("segments").glob("*.json"))

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(process_segment_wrapper, seg)
        for seg in segments
    ]

    for future in as_completed(futures):
        success, segment_path, result = future.result()
        if success:
            print(f"✓ {segment_path.name} -> {result}")
        else:
            print(f"✗ {segment_path.name}: {result}")
```

## Integration Examples

### Example 16: Creating Segment Metadata from Scratch

```python
from auto_annotator.adapters import InputAdapter

# Create segment metadata programmatically
metadata_dict = {
    "segment_id": "custom_001",
    "original_video": {
        "path": "Dataset/Archery/Men/1.mp4",
        "json_path": "Dataset/Archery/Men/1.json",
        "sport": "Archery",
        "event": "Men",
        "video_id": "1"
    },
    "segment_info": {
        "path": "Dataset/Archery/Men/segments/1_001.mp4",
        "start_frame_in_original": 0,
        "total_frames": 150,
        "fps": 10,
        "duration_sec": 15.0,
        "resolution": [1920, 1080]
    },
    "tasks_to_annotate": [
        "Spatial_Temporal_Grounding",
        "Continuous_Actions_Caption"
    ],
    "additional_info": {
        "description": "Archer preparing and shooting arrow"
    }
}

metadata = InputAdapter.create_from_dict(metadata_dict)

# Save to file
from auto_annotator.utils import JSONUtils
JSONUtils.save_json(
    metadata_dict,
    Path("segments/custom_001.json")
)
```

### Example 17: Command Line Automation

```bash
#!/bin/bash
# Process all sports categories

SPORTS=("3x3_Basketball" "Archery" "Swimming")
EVENTS=("Men" "Women")

for sport in "${SPORTS[@]}"; do
    for event in "${EVENTS[@]}"; do
        echo "Processing $sport - $event"
        uv run python -m auto_annotator.main \
            "segments/$sport/$event/" \
            -o "output/temp/$sport/$event/" \
            -v
    done
done

echo "All processing complete!"
```

## Testing and Validation

### Example 18: Unit Testing Custom Functions

```python
import pytest
from auto_annotator.utils import JSONUtils

def test_custom_merge_logic():
    """Test custom annotation merging."""
    base = {
        "annotations": [
            {"annotation_id": "1", "task": "A"}
        ]
    }

    new = [
        {"task": "B"},
        {"task": "C"}
    ]

    result = JSONUtils.merge_annotations(base, new)

    assert len(result["annotations"]) == 3
    assert result["annotations"][1]["annotation_id"] == "2"
    assert result["annotations"][2]["annotation_id"] == "3"

# Run with: uv run pytest test_custom.py
```

These examples cover the main use cases for AutoAnnotator. For more information, see the [README.md](../README.md) and source code documentation.
