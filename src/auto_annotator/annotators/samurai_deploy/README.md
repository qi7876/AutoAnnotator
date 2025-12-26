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

## AutoAnnotator Tracker 配置

AutoAnnotator 支持自定义权重路径与自动下载（HuggingFace）：

在 `config/config.yaml` 中配置：

```yaml
tasks:
  tracking:
    tracker_backend: "local"
    model_path: "/abs/path/to/sam2.1_hiera_base_plus.pt"
    hf_model_id: "facebook/sam2.1-hiera-base-plus"
    auto_download: false
```

说明：
- `model_path` 指向本地权重文件（存在则优先使用）。
- 当 `model_path` 不存在且 `auto_download=true` 时，会使用 `hf_model_id` 自动下载权重。
- 未提供 `hf_model_id` 或未开启 `auto_download` 时，将回退到静态 bbox。

## Example

```bash
python demo.py --video_path test/demo_video.mp4 --json_path test/bboxes.json
```

This will process the demo video using the provided bounding boxes and generate tracking results.
