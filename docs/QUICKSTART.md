# Quick Start Guide

Get up and running with AutoAnnotator in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) installed
- Google AI Studio API key ([Get one here](https://aistudio.google.com/apikey))

## Step 1: Setup

```bash
# Navigate to project directory
cd /path/to/AutoAnnotator

# Install dependencies
uv sync

# Setup environment variables
cp config/.env.example config/.env
```

Edit `config/.env` and add your API key:
```env
GEMINI_API_KEY=your_api_key_here
```

## Step 2: Prepare Your Data

Create a segment metadata JSON file (e.g., `test_segment.json`):

```json
{
  "segment_id": "test_001",
  "original_video": {
    "path": "path/to/original/video.mp4",
    "json_path": "path/to/original/video.json",
    "sport": "Basketball",
    "event": "Men",
    "video_id": "1"
  },
  "segment_info": {
    "path": "path/to/segment.mp4",
    "start_frame_in_original": 100,
    "total_frames": 50,
    "fps": 10,
    "duration_sec": 5.0,
    "resolution": [1920, 1080]
  },
  "tasks_to_annotate": [
    "ScoreboardSingle"
  ]
}
```

## Step 3: Run Annotation

```bash
# Process single segment
uv run python -m auto_annotator.main test_segment.json

# Or process a directory of segments
uv run python -m auto_annotator.main segments/
```

## Step 4: Check Results

Results are saved in `output/temp/` by default:

```bash
# View annotation results
cat output/temp/test_001.json
```

## Next Steps

- Read the [full documentation](../README.md)
- See [usage examples](USAGE_EXAMPLES.md)
- Explore the [segment metadata schema](segment_metadata_schema.json)
- Customize [configuration](../config/config.yaml)
- Implement [bounding box annotation](../src/auto_annotator/annotators/bbox_annotator.py)
- Implement [object tracking](../src/auto_annotator/annotators/tracker.py)

## Common Issues

### "GEMINI_API_KEY not found"

Make sure you created `config/.env` with your API key:
```bash
cp config/.env.example config/.env
# Edit config/.env and add your key
```

### "Video file not found"

Check that the video paths in your segment metadata JSON are correct and relative to your current directory or use absolute paths.

### Import errors

Reinstall dependencies:
```bash
uv sync --reinstall
```

## Getting Help

- Check the [troubleshooting section](../README.md#troubleshooting) in README
- Review [usage examples](USAGE_EXAMPLES.md)
- Open an issue on GitHub (if applicable)

## What's Next?

Now that you have basic annotation working, you can:

1. **Customize prompts**: Edit files in `config/prompts/` to improve annotation quality
2. **Implement bounding box detection**: Complete the `BBoxAnnotator` interface
3. **Add object tracking**: Implement the `ObjectTracker` interface
4. **Batch process**: Annotate your entire dataset
5. **Review and merge**: Use Step 4 (human review) and Step 5 (JSON merging) in your workflow

Happy annotating! 🎯
