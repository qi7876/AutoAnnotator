# 数据集结构说明

## 数据集目录结构

```
Dataset/
└── {Sport}/                    # 运动项目名称（如：Archery, 3x3_Basketball）
    └── {Event}/                # 比赛事件名称（如：Men's_Individual, Men, Women's_Team）
        ├── {video_id}.mp4      # 原始视频文件（如：1.mp4, 2.mp4, 3.mp4）
        ├── {video_id}.json     # 原始视频的元数据
        ├── metainfo.json       # 事件级别的元信息
        ├── segment_dir/        # 视频片段目录
        │   ├── {segment_id}.mp4
        │   └── {segment_id}.json
        └── singleframes_dir/   # 单帧图片目录
            ├── {segment_id}.jpg
            └── {segment_id}.json
```

## 示例：3x3_Basketball 和 Archery 项目

```
3x3_Basketball/
└── Men/
    ├── 1.json
    ├── 1.mp4
    ├── 7.json
    ├── 7.mp4
    ├── metainfo.json
    ├── segment_dir/
    │   ├── 1_split_7_start_000652.json
    │   └── 1_split_7_start_000652.mp4
    └── singleframes_dir/
        ├── 1_frame_119.jpg
        └── 1_frame_119.json

Archery/
├── Men's_Individual/
│   ├── 1.json
│   ├── 1.mp4
│   ├── metainfo.json
│   ├── segment_dir/
│   │   ├── 1_split_1_start_000292.json
│   │   └── 1_split_1_start_000292.mp4
│   └── singleframes_dir/
│       ├── 5.jpg
│       ├── 5.json
│       ├── 1_frame_3401.jpg
│       └── 1_frame_3401.json
```

## 元数据格式

### 视频片段元数据 (segment_dir/)

```json
{
  "segment_id": "1_split_7_start_000652",
  "original_video": {
    "sport": "3x3_Basketball",
    "event": "Men"
  },
  "segment_info": {
    "start_frame_in_original": 6520,
    "total_frames": 70,
    "fps": 10.0,
    "duration_sec": 7.0,
    "resolution": [1920, 1080]
  },
  "tasks_to_annotate": [
    "Continuous_Actions_Caption"
  ],
  "additional_info": {
    "description": "Video segment from frame 6520 with 70 frames"
  }
}
```

### 单帧元数据 (singleframes_dir/)

```json
{
  "segment_id": 5,
  "original_video": {
    "sport": "Archery",
    "event": "Men's_Individual"
  },
  "segment_info": {
    "start_frame_in_original": 7462,
    "total_frames": 1,
    "fps": 10.0,
    "duration_sec": 0.1,
    "resolution": [1920, 1080]
  },
  "tasks_to_annotate": [
    "ScoreboardSingle"
  ],
  "additional_info": {
    "description": "Extracted single frame at time 746.200s."
  }
}
```

## 关键特性

### 1. 统一的元数据格式

片段和单帧使用**相同的元数据格式**，区分方式：
- **片段**: `total_frames > 1`
- **单帧**: `total_frames == 1`

### 2. segment_id 格式

#### 视频片段
格式: `{video_id}_split_{split_num}_start_{start_frame:06d}`

示例:
- `1_split_7_start_000652` - 来自视频1，第7个片段，从帧652开始
- `3_split_2_start_001234` - 来自视频3，第2个片段，从帧1234开始

#### 单帧
可以是以下两种格式之一：
- **整数**: `5`, `123` （简单ID）
- **字符串**: `1_frame_3401` （来自视频1，帧号3401）

### 3. video_id 自动提取

`video_id` **不存储在 JSON 中**，而是从 `segment_id` 自动提取：
- `1_split_7_start_000652` → video_id = 1
- `3_split_2_start_001234` → video_id = 3
- `1_frame_3401` → video_id = 1
- 整数 segment_id → video_id = 1 (默认)

## 路径构造规则

### 原始视频
- **视频文件**: `Dataset/{sport}/{event}/{video_id}.mp4`
- **元数据文件**: `Dataset/{sport}/{event}/{video_id}.json`
- **事件元信息**: `Dataset/{sport}/{event}/metainfo.json`

### 视频片段
- **视频文件**: `Dataset/{sport}/{event}/segment_dir/{segment_id}.mp4`
- **元数据文件**: `Dataset/{sport}/{event}/segment_dir/{segment_id}.json`

### 单帧图片
- **图片文件**: `Dataset/{sport}/{event}/singleframes_dir/{segment_id}.jpg`
- **元数据文件**: `Dataset/{sport}/{event}/singleframes_dir/{segment_id}.json`

## 使用示例

```python
from pathlib import Path
from auto_annotator.adapters import InputAdapter

# 加载单帧元数据（整数 ID）
metadata = InputAdapter.load_from_json(
    Path("Dataset/Archery/Men's_Individual/singleframes_dir/5.json")
)

# 获取图片路径
image_path = metadata.get_video_path(Path("Dataset"))
# 返回: Dataset/Archery/Men's_Individual/singleframes_dir/5.jpg

# 获取原始视频路径（video_id 自动从 segment_id 提取）
original_video = metadata.get_original_video_path(Path("Dataset"))
# 返回: Dataset/Archery/Men's_Individual/1.mp4

# 检查类型
if metadata.segment_info.is_single_frame():
    print("This is a single frame")
elif metadata.segment_info.is_segment():
    print("This is a video segment")

# 加载片段元数据
segment = InputAdapter.load_from_json(
    Path("Dataset/3x3_Basketball/Men/segment_dir/1_split_7_start_000652.json")
)

# 获取片段视频路径
video_path = segment.get_video_path(Path("Dataset"))
# 返回: Dataset/3x3_Basketball/Men/segment_dir/1_split_7_start_000652.mp4

# video_id 自动提取
original_video = segment.get_original_video_path(Path("Dataset"))
# 返回: Dataset/3x3_Basketball/Men/1.mp4

# 加载事件目录下的所有元数据
all_metadata = InputAdapter.load_from_event_directory(
    Path("Dataset/Archery/Men's_Individual")
)

# 只加载单帧元数据
singleframe_metadata = InputAdapter.load_from_event_directory(
    Path("Dataset/Archery/Men's_Individual"),
    single_frame_only=True
)
```

## 字段说明

### segment_id
- **类型**: 字符串或整数
- **格式**:
  - 片段: `"{video_id}_split_{split_num}_start_{start_frame:06d}"`
  - 单帧: 整数（如 `5`）或字符串（如 `"1_frame_3401"`）

### original_video
- `sport`: 运动项目名称
- `event`: 比赛事件名称
- **不包含** `video_id` 字段（从 segment_id 提取）

### segment_info
- `start_frame_in_original`: 在原始视频中的起始帧号
- `total_frames`: 帧数（**1 = 单帧，>1 = 片段**）
- `fps`: 帧率（浮点数）
- `duration_sec`: 时长（秒）
- `resolution`: 分辨率 [宽度, 高度]

### 通用字段
- `tasks_to_annotate`: 要执行的标注任务列表
- `additional_info`: 附加元数据信息

## 设计优势

1. **格式统一**: 片段和单帧使用相同的 JSON 结构
2. **避免冗余**: video_id 不单独存储，从 segment_id 提取
3. **类型灵活**: segment_id 支持字符串和整数
4. **简单判断**: 通过 `total_frames` 区分类型，无需额外字段
5. **易于扩展**: 支持多个原始视频（通过 video_id）
