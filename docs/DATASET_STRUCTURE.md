# 数据集结构说明

## 数据集目录结构

```
data/Dataset/
└── {Sport}/                    # 运动项目名称（如：Archery, 3x3_Basketball）
    └── {Event}/                # 比赛事件名称（如：Men's_Individual, Men, Women's_Team）
        ├── {video_id}.mp4      # 原始视频文件（如：1.mp4, 2.mp4, 3.mp4）
        ├── {video_id}.json     # 原始视频的元数据
        ├── metainfo.json       # 事件级别的元信息
        ├── clips/              # 视频片段目录
        │   ├── {id}.mp4
        │   └── {id}.json
        └── frames/             # 单帧图片目录
            ├── {id}.jpg
            └── {id}.json
```

## 示例：Archery 项目

```
data/Dataset/
└── Archery/
    └── Men's_Individual/
        ├── 1.json
        ├── 1.mp4
        ├── 2.json
        ├── 2.mp4
        ├── 3.json
        ├── 3.mp4
        ├── 4.json
        ├── 4.mp4
        ├── metainfo.json
        ├── clips/              # 视频片段
        │   ├── 1.json
        │   ├── 1.mp4
        │   ├── 2.json
        │   └── 2.mp4
        └── frames/             # 单帧图片
            ├── 1.jpg
            ├── 1.json
            ├── 2.jpg
            ├── 2.json
            ├── 3.jpg
            ├── 3.json
            ├── 4.jpg
            └── 4.json
```

## 元数据格式

### 视频片段元数据 (clips/)

```json
{
  "id": "1",
  "origin": {
    "sport": "3x3_Basketball",
    "event": "Men"
  },
  "info": {
    "original_starting_frame": 6520,
    "total_frames": 70,
    "fps": 10.0
  },
  "tasks_to_annotate": [
    "Continuous_Actions_Caption"
  ]
}
```

### 单帧元数据 (frames/)

```json
{
  "id": "1",
  "origin": {
    "sport": "Archery",
    "event": "Men's_Individual"
  },
  "info": {
    "original_starting_frame": 7462,
    "total_frames": 1,
    "fps": 10.0
  },
  "tasks_to_annotate": [
    "ScoreboardSingle"
  ]
}
```

## 关键特性

### 1. 统一的元数据格式

片段和单帧使用**相同的元数据格式**，区分方式：
- **片段**: `total_frames > 1`
- **单帧**: `total_frames == 1`

### 2. 简化的 ID 格式

所有片段和单帧都使用**简单的字符串 ID**：
- 片段: `"1"`, `"2"`, `"3"` 等
- 单帧: `"1"`, `"2"`, `"3"` 等

ID 与文件名一致（不包含扩展名）。

### 3. 简化的字段结构

- `id`: 唯一标识符，与文件名匹配
- `origin`: 包含 `sport` 和 `event` 的原始视频来源信息
- `info`: 包含 `original_starting_frame`, `total_frames`, `fps` 的片段/帧信息
- `tasks_to_annotate`: 要执行的标注任务列表

## 路径构造规则

### 原始视频
- **视频文件**: `data/Dataset/{sport}/{event}/{video_id}.mp4`
- **元数据文件**: `data/Dataset/{sport}/{event}/{video_id}.json`
- **事件元信息**: `data/Dataset/{sport}/{event}/metainfo.json`

### 视频片段
- **视频文件**: `data/Dataset/{sport}/{event}/clips/{id}.mp4`
- **元数据文件**: `data/Dataset/{sport}/{event}/clips/{id}.json`

### 单帧图片
- **图片文件**: `data/Dataset/{sport}/{event}/frames/{id}.jpg`
- **元数据文件**: `data/Dataset/{sport}/{event}/frames/{id}.json`

## 使用示例

```python
from pathlib import Path
from auto_annotator.adapters import InputAdapter

# 加载单帧元数据
metadata = InputAdapter.load_from_json(
    Path("data/Dataset/Archery/Men's_Individual/frames/1.json")
)

# 获取图片路径
image_path = metadata.get_video_path(Path("Dataset"))
# 返回: data/Dataset/Archery/Men's_Individual/frames/1.jpg

# 获取原始视频路径
original_video = metadata.get_original_video_path(Path("Dataset"), video_id="1")
# 返回: data/Dataset/Archery/Men's_Individual/1.mp4

# 检查类型
if metadata.info.is_single_frame():
    print("This is a single frame")
elif metadata.info.is_clip():
    print("This is a video clip")

# 加载片段元数据
clip = InputAdapter.load_from_json(
    Path("data/Dataset/3x3_Basketball/Men/clips/1.json")
)

# 获取片段视频路径
video_path = clip.get_video_path(Path("Dataset"))
# 返回: data/Dataset/3x3_Basketball/Men/clips/1.mp4

# 加载事件目录下的所有元数据
all_metadata = InputAdapter.load_from_event_directory(
    Path("data/Dataset/Archery/Men's_Individual")
)

# 只加载单帧元数据
singleframe_metadata = InputAdapter.load_from_event_directory(
    Path("data/Dataset/Archery/Men's_Individual"),
    single_frame_only=True
)
```

## 字段说明

### id
- **类型**: 字符串
- **说明**: 唯一标识符，与文件名匹配（不包含扩展名）
- **示例**: `"1"`, `"2"`, `"3"`

### origin
- `sport`: 运动项目名称（字符串）
- `event`: 比赛事件名称（字符串）

### info
- `original_starting_frame`: 在原始视频中的起始帧号（整数）
- `total_frames`: 帧数（**1 = 单帧，>1 = 片段**）
- `fps`: 帧率（浮点数）

### tasks_to_annotate
- **类型**: 字符串列表
- **说明**: 要执行的标注任务列表
- **示例**: `["UCE"]`, `["ScoreboardSingle"]`
