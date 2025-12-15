# SAMURAI Deployment Package

This is a minimal deployment package for running SAMURAI video object tracking.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run demo:
```bash
python demo.py
```

## Files Structure

- `demo.py`: Main demo script for video object tracking
- `sam2/`: SAM2 model files and configurations 
- `sam2.1_hiera_base_plus.pt`: Pre-trained model checkpoint
- `test/`: Test data including video and bounding box annotations
- `requirements.txt`: Python dependencies

## Usage

The demo script accepts the following parameters:

- `--video_path`: Input video path or directory of frames (default: test/demo_video.mp4)
- `--json_path`: Path to bounding box JSON file (default: test/bboxes.json)  
- `--model_path`: Path to model checkpoint (default: sam2.1_hiera_base_plus.pt)
- `--video_output_path`: Output video path (default: test/annotated_deploy.mp4)
- `--results_path`: Results JSON path (default: test/results_deploy.json)
- `--save_to_video`: Whether to save video output (default: True)

## Example

```bash
python demo.py --video_path test/demo_video.mp4 --json_path test/bboxes.json
```

This will process the demo video using the provided bounding boxes and generate tracking results.