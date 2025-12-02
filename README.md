# AutoAnnotator

AI-powered video annotation system for multimodal sports datasets using Google's Gemini API.

## Overview

AutoAnnotator is a comprehensive Python toolkit designed to automatically annotate sports videos for multimodal benchmark datasets. It leverages Google's Gemini API to perform 7 different annotation tasks, covering both perception and understanding levels.

## Features

- **7 Annotation Tasks**:
  - Scoreboard Understanding (Single Frame)
  - Scoreboard Understanding (Multiple Frames)
  - Objects Spatial Relationships
  - Spatial-Temporal Grounding
  - Continuous Actions Caption
  - Continuous Events Caption
  - Object Tracking

- **Modular Architecture**:
  - Decoupled input adapter for flexibility
  - Task-specific annotators with unified interface
  - Extensible bounding box and tracking interfaces
  - Comprehensive configuration management

- **Production Ready**:
  - Robust error handling and logging
  - JSON validation and merging utilities
  - Batch processing support
  - Temporary output management

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Google AI Studio API key

### Setup

1. Clone the repository:
```bash
cd /path/to/AutoAnnotator
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Configure API key:
```bash
cp config/.env.example config/.env
# Edit config/.env and add your GEMINI_API_KEY
```

4. Verify installation:
```bash
uv run python -c "from auto_annotator import get_config; print('Setup complete!')"
```

## Configuration

### Environment Variables

Create a `config/.env` file:

```env
GEMINI_API_KEY=your_api_key_here
PROJECT_ROOT=/path/to/AutoAnnotator
DATASET_ROOT=/path/to/Dataset
```

### Configuration File

Edit `config/config.yaml` to customize:

- Gemini model settings
- Output directories
- Task-specific parameters
- Logging configuration

## Usage

### Command Line Interface

Process a single segment:
```bash
uv run python -m auto_annotator.main path/to/segment_metadata.json
```

Process multiple segments in a directory:
```bash
uv run python -m auto_annotator.main path/to/segments_dir/
```

Specify custom output directory:
```bash
uv run python -m auto_annotator.main path/to/segments/ -o output/custom/
```

Enable verbose logging:
```bash
uv run python -m auto_annotator.main path/to/segments/ -v
```

### Python API

```python
from pathlib import Path
from auto_annotator import (
    InputAdapter,
    GeminiClient,
    TaskAnnotatorFactory,
    PromptLoader
)
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker

# Initialize components
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator()
tracker = ObjectTracker()

# Load segment metadata
segment_metadata = InputAdapter.load_from_json(
    Path("segments/segment_001.json")
)

# Create annotator for a specific task
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="ScoreboardSingle",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

# Perform annotation
annotation = annotator.annotate(segment_metadata)
print(annotation)
```

## Segment Metadata Format

Input segments should follow this JSON format:

```json
{
  "segment_id": "1_segment_001",
  "original_video": {
    "path": "Dataset/3x3_Basketball/Men/1.mp4",
    "json_path": "Dataset/3x3_Basketball/Men/1.json",
    "sport": "3x3_Basketball",
    "event": "Men",
    "video_id": "1"
  },
  "segment_info": {
    "path": "Dataset/3x3_Basketball/Men/segments/1_segment_001.mp4",
    "start_frame_in_original": 150,
    "total_frames": 100,
    "fps": 10,
    "duration_sec": 10.0,
    "resolution": [1920, 1080]
  },
  "tasks_to_annotate": [
    "ScoreboardSingle",
    "Continuous_Actions_Caption"
  ]
}
```

See [docs/segment_metadata_schema.json](docs/segment_metadata_schema.json) for the complete schema.

## Output Format

Annotations are saved as JSON files in the temporary output directory:

```json
{
  "segment_id": "1_segment_001",
  "original_video": {
    "sport": "3x3_Basketball",
    "event": "Men",
    "video_id": "1"
  },
  "annotations": [
    {
      "annotation_id": "1",
      "task_L1": "Understanding",
      "task_L2": "ScoreboardSingle",
      "timestamp_frame": 50,
      "question": "Based on the scoreboard, who is in first place?",
      "answer": "The Lakers are in first place.",
      "bounding_box": [934, 452, 1041, 667]
    }
  ]
}
```

## Development

### Project Structure

```
AutoAnnotator/
├── config/
│   ├── prompts/           # Task prompt templates
│   ├── config.yaml        # Main configuration
│   └── .env.example       # Environment variables template
├── src/auto_annotator/
│   ├── adapters/          # Input format adapters
│   ├── annotators/        # Task annotators and AI clients
│   ├── utils/             # Utility modules
│   ├── config.py          # Configuration management
│   └── main.py            # Main entry point
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── output/
    ├── temp/              # Temporary annotations
    └── final/             # Final merged annotations
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_config.py

# Run with coverage
uv run pytest --cov=auto_annotator
```

### Adding New Tasks

1. Create prompt template in `config/prompts/`
2. Implement annotator class inheriting from `BaseAnnotator`
3. Register in `TaskAnnotatorFactory`
4. Update configuration and documentation

## Extending Functionality

### Implementing Bounding Box Annotation

The bounding box annotation interface is defined in [src/auto_annotator/annotators/bbox_annotator.py](src/auto_annotator/annotators/bbox_annotator.py).

To implement:
1. Complete the `annotate_single_object()` method
2. Complete the `annotate_multiple_objects()` method
3. Implement frame extraction if needed

Example using Gemini grounding:
```python
def annotate_single_object(self, image, description):
    # Use Gemini grounding model
    response = grounding_model.generate_content([image, description])
    # Parse bounding box from response
    bbox = BoundingBox(xtl, ytl, xbr, ybr)
    return bbox
```

### Implementing Object Tracking

The object tracking interface is defined in [src/auto_annotator/annotators/tracker.py](src/auto_annotator/annotators/tracker.py).

To implement:
1. Choose tracking backend (ByteTrack, DeepSORT, etc.)
2. Implement `track_from_first_bbox()` method
3. Implement `track_with_query()` method

Example structure:
```python
def track_from_first_bbox(self, video_path, first_bbox, start_frame, end_frame):
    # Initialize tracker
    tracker = YourTracker()

    # Track object across frames
    bboxes = []
    for frame_num in range(start_frame, end_frame + 1):
        bbox = tracker.update(frame)
        bboxes.append(bbox)

    return TrackingResult(video_path, start_frame, end_frame, bboxes)
```

## Troubleshooting

### API Key Issues

If you see "GEMINI_API_KEY not found":
1. Ensure `config/.env` exists
2. Check that the API key is correctly set
3. Verify the API key is valid in Google AI Studio

### Video Upload Timeout

If videos are timing out during upload:
1. Increase `upload_timeout_sec` in `config/config.yaml`
2. Check your internet connection
3. Consider splitting large videos into smaller segments

### Import Errors

If you encounter import errors:
```bash
# Reinstall dependencies
uv sync --reinstall

# Check Python version
python --version  # Should be 3.10+
```

## Workflow Integration

AutoAnnotator is designed as Step 3 in a 5-step annotation pipeline:

1. **Segment Splitting**: Extract relevant clips from full videos
2. **Human Review**: Verify segment quality
3. **AI Annotation** (AutoAnnotator): Generate annotations
4. **Human Review**: Verify and correct annotations
5. **JSON Merging**: Combine into final dataset

## Contributing

This project is part of a multimodal benchmark research effort. For collaboration inquiries, please contact the project maintainer.

## License

MIT

## Acknowledgments

- Google Gemini API for multimodal understanding
- OpenCV for video processing
- uv for modern Python package management
