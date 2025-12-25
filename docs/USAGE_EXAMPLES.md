# 使用示例

本文档提供 AutoAnnotator 在各种场景下的详细使用示例。

## 📚 目录

- [基础使用](#基础使用)
- [任务专用示例](#任务专用示例)
- [高级用法](#高级用法)
- [批量处理](#批量处理)
- [自定义配置](#自定义配置)

## 基础使用

### 示例 1：标注单个片段

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

# 初始化配置和组件
config = get_config()
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()
output_dir = Path("data/output/temp")

# 加载片段元数据
segment_metadata = InputAdapter.load_from_json(
    Path("data/Dataset/Archery/Men's_Individual/singleframes_dir/5.json")
)

# 处理标注
output_path = process_segment(
    segment_metadata=segment_metadata,
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker,
    output_dir=output_dir,
    dataset_root=config.dataset_root
)

print(f"标注结果已保存到: {output_path}")
```

### 示例 2：批量处理多个片段

```python
from pathlib import Path
from auto_annotator.main import process_segments_batch

# 处理目录中的所有片段
segments_dir = Path("data/Dataset/3x3_Basketball/Men/segment_dir")
output_dir = Path("data/output/temp")

# 批量处理
process_segments_batch(
    segment_paths=list(segments_dir.glob("*.json")),
    output_dir=output_dir
)

print("批量标注完成！")
```

增量更新说明：
- 当输出目录中已存在 `{clip_id}.json` 时，该片段会被自动跳过。
- 当以目录形式输入时，会自动删除输出目录中“源元数据已不存在”的标注结果。
- 若已存在的标注缺少任务，将仅补标缺失任务。

### 示例 3：只处理特定任务

```python
from pathlib import Path
from auto_annotator import InputAdapter

# 加载片段元数据
segment_metadata = InputAdapter.load_from_json(
    Path("data/Dataset/Archery/Men's_Individual/segment_dir/1_split_1_start_000292.json")
)

# 只标注计分板理解任务
segment_metadata.tasks_to_annotate = [
    "ScoreboardSingle"
]

# 然后正常处理...
```

## 任务专用示例

### 示例 4：计分板理解（单帧）

```python
from pathlib import Path
from auto_annotator import (
    TaskAnnotatorFactory,
    GeminiClient,
    PromptLoader,
    InputAdapter
)
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.config import get_config

# 初始化
config = get_config()
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()

# 创建计分板单帧标注器
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="ScoreboardSingle",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

# 加载单帧元数据
segment_metadata = InputAdapter.load_from_json(
    Path("data/Dataset/Archery/Men's_Individual/singleframes_dir/5.json")
)

# 执行标注
annotation = annotator.annotate(
    segment_metadata,
    dataset_root=config.dataset_root
)

print("标注结果:")
print(f"  任务: {annotation['task_L2']}")
print(f"  问题: {annotation['question']}")
print(f"  答案: {annotation['answer']}")
if 'bounding_box' in annotation:
    print(f"  边界框: {annotation['bounding_box']}")
```

### 示例 5：连续动作描述

```python
from pathlib import Path
from auto_annotator import (
    TaskAnnotatorFactory,
    GeminiClient,
    PromptLoader,
    InputAdapter
)
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.config import get_config

# 初始化
config = get_config()
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()

# 创建连续动作描述标注器
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="Continuous_Actions_Caption",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

# 加载视频片段元数据
segment_metadata = InputAdapter.load_from_json(
    Path("data/Dataset/3x3_Basketball/Men/segment_dir/1_split_7_start_000652.json")
)

# 执行标注
annotation = annotator.annotate(
    segment_metadata,
    dataset_root=config.dataset_root
)

print("动作描述标注结果:")
print(f"  任务: {annotation['task_L2']}")
print(f"  问题: {annotation['question']}")
print(f"  答案: {annotation['answer']}")
```

### 示例 6：物体空间关系

```python
from pathlib import Path
from auto_annotator import (
    TaskAnnotatorFactory,
    GeminiClient,
    PromptLoader,
    InputAdapter
)
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.config import get_config

# 初始化
config = get_config()
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()

# 创建物体空间关系标注器
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="Objects_Spatial_Relationships",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

# 加载单帧元数据
segment_metadata = InputAdapter.load_from_json(
    Path("data/Dataset/Archery/Men's_Individual/singleframes_dir/5.json")
)

# 执行标注
annotation = annotator.annotate(
    segment_metadata,
    dataset_root=config.dataset_root
)

print("物体空间关系标注结果:")
for item in annotation.get('spatial_relationships', []):
    print(f"  - {item}")
```

## 高级用法

### 示例 7：加载并验证元数据

```python
from pathlib import Path
from auto_annotator import InputAdapter
from auto_annotator.config import get_config

config = get_config()

# 加载元数据
metadata_path = Path("data/Dataset/Archery/Men's_Individual/frames/1.json")
clip_metadata = InputAdapter.load_from_json(metadata_path)

# 检查类型
if clip_metadata.info.is_single_frame():
    print("✓ 这是单帧图片")
    print(f"  帧号: {clip_metadata.info.original_starting_frame}")
elif clip_metadata.info.is_clip():
    print("✓ 这是视频片段")
    print(f"  总帧数: {clip_metadata.info.total_frames}")

# 获取路径信息
content_path = clip_metadata.get_video_path(config.dataset_root)
original_video = clip_metadata.get_original_video_path(config.dataset_root)

print(f"内容路径: {content_path}")
print(f"原始视频: {original_video}")

# 验证元数据
is_valid, error = InputAdapter.validate_metadata(
    clip_metadata,
    dataset_root=config.dataset_root,
    check_file_existence=True
)

if is_valid:
    print("✓ 元数据验证通过")
else:
    print(f"✗ 元数据验证失败: {error}")
```

### 示例 8：从事件目录加载所有元数据

```python
from pathlib import Path
from auto_annotator import InputAdapter

# 加载事件目录下的所有元数据
event_dir = Path("data/Dataset/Archery/Men's_Individual")
all_metadata = InputAdapter.load_from_event_directory(event_dir)

print(f"找到 {len(all_metadata)} 个片段/单帧")

# 统计类型
clips = [m for m in all_metadata if m.info.is_clip()]
frames = [m for m in all_metadata if m.info.is_single_frame()]

print(f"  视频片段: {len(clips)}")
print(f"  单帧图片: {len(frames)}")

# 只加载单帧
singleframes_only = InputAdapter.load_from_event_directory(
    event_dir,
    single_frame_only=True
)
print(f"只加载单帧: {len(singleframes_only)} 个")
```

### 示例 9：自定义输出目录

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

config = get_config()

# 初始化组件
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()

# 自定义输出目录（按日期）
from datetime import datetime
output_dir = Path(f"data/output/annotations_{datetime.now().strftime('%Y%m%d')}")
output_dir.mkdir(parents=True, exist_ok=True)

# 加载并处理
segment_metadata = InputAdapter.load_from_json(
    Path("data/Dataset/Archery/Men's_Individual/singleframes_dir/5.json")
)

output_path = process_segment(
    segment_metadata=segment_metadata,
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker,
    output_dir=output_dir,
    dataset_root=config.dataset_root
)

print(f"标注结果保存到: {output_path}")
```

### 示例 10：错误处理

```python
from pathlib import Path
from auto_annotator import InputAdapter, GeminiClient
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_annotate(metadata_path: Path):
    """安全地执行标注，处理可能的错误"""
    try:
        # 加载元数据
        segment_metadata = InputAdapter.load_from_json(metadata_path)
        logger.info(f"成功加载元数据: {metadata_path}")

        # 验证
        is_valid, error = InputAdapter.validate_metadata(
            segment_metadata,
            check_file_existence=True
        )

        if not is_valid:
            logger.error(f"元数据验证失败: {error}")
            return None

        # 初始化客户端
        gemini_client = GeminiClient()

        # 执行标注
        # ... 标注逻辑 ...

        logger.info("标注成功完成")
        return True

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        return None
    except ValueError as e:
        logger.error(f"值错误: {e}")
        return None
    except Exception as e:
        logger.error(f"未预期的错误: {e}", exc_info=True)
        return None

# 使用
result = safe_annotate(
    Path("data/Dataset/Archery/Men's_Individual/singleframes_dir/5.json")
)
```

## 批量处理

### 示例 11：批量处理多个运动项目

```python
from pathlib import Path
from auto_annotator import InputAdapter
from auto_annotator.main import process_segment
from auto_annotator import GeminiClient, PromptLoader
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.config import get_config

config = get_config()
dataset_root = Path(config.dataset_root)

# 初始化组件
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()

# 遍历所有运动项目
for sport_dir in dataset_root.iterdir():
    if not sport_dir.is_dir():
        continue

    print(f"\n处理运动项目: {sport_dir.name}")

    # 遍历所有比赛事件
    for event_dir in sport_dir.iterdir():
        if not event_dir.is_dir():
            continue

        print(f"  处理事件: {event_dir.name}")

        # 加载所有元数据
        metadata_list = InputAdapter.load_from_event_directory(event_dir)

        print(f"    找到 {len(metadata_list)} 个片段/单帧")

        # 处理每个片段
        for metadata in metadata_list:
            try:
                output_path = process_segment(
                    segment_metadata=metadata,
                    gemini_client=gemini_client,
                    prompt_loader=prompt_loader,
                    bbox_annotator=bbox_annotator,
                    tracker=tracker,
                    output_dir=Path("data/output/temp"),
                    dataset_root=config.dataset_root
                )
                print(f"      ✓ {metadata.id}")
            except Exception as e:
                print(f"      ✗ {metadata.id}: {e}")

print("\n批量处理完成！")
```

### 示例 12：并行批量处理

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from auto_annotator import InputAdapter
from auto_annotator.main import process_segment
from auto_annotator import GeminiClient, PromptLoader
from auto_annotator.annotators.bbox_annotator import BBoxAnnotator
from auto_annotator.annotators.tracker import ObjectTracker
from auto_annotator.config import get_config

config = get_config()

def process_single_metadata(metadata_path: Path):
    """处理单个元数据文件"""
    try:
        # 为每个线程创建独立的客户端
        gemini_client = GeminiClient()
        prompt_loader = PromptLoader()
        bbox_annotator = BBoxAnnotator(gemini_client)
        tracker = ObjectTracker()

        # 加载和处理
        segment_metadata = InputAdapter.load_from_json(metadata_path)
        output_path = process_segment(
            segment_metadata=segment_metadata,
            gemini_client=gemini_client,
            prompt_loader=prompt_loader,
            bbox_annotator=bbox_annotator,
            tracker=tracker,
            output_dir=Path("data/output/temp"),
            dataset_root=config.dataset_root
        )
        return (metadata_path.name, True, None)
    except Exception as e:
        return (metadata_path.name, False, str(e))

# 收集所有元数据文件
event_dir = Path("data/Dataset/Archery/Men's_Individual")
all_json_files = list(event_dir.glob("**/*.json"))
all_json_files = [f for f in all_json_files if not f.name.startswith("annotation_")]

print(f"找到 {len(all_json_files)} 个元数据文件")

# 并行处理（注意：控制并发数以避免 API 限流）
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_single_metadata, f) for f in all_json_files]

    for future in as_completed(futures):
        filename, success, error = future.result()
        if success:
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename}: {error}")

print("\n并行批量处理完成！")
```

## 自定义配置

### 示例 13：使用自定义提示词

```python
from pathlib import Path
from auto_annotator import PromptLoader, GeminiClient

# 创建自定义提示词加载器
prompt_loader = PromptLoader(prompts_dir=Path("my_custom_prompts"))

# 使用自定义提示词
gemini_client = GeminiClient()
custom_prompt = prompt_loader.load_prompt("my_custom_task")

# 使用提示词进行标注
response = gemini_client.generate_content([video, custom_prompt])
```

### 示例 14：调整 API 参数

```python
from auto_annotator.config import get_config

# 获取配置
config = get_config()

# 显示当前配置
print(f"Gemini 模型: {config.gemini.model}")
print(f"上传超时: {config.gemini.video['upload_timeout_sec']}秒")
print(f"处理超时: {config.gemini.video['processing_timeout_sec']}秒")

# 可以在 config/config.yaml 中修改这些参数
```

## 📚 更多资源

- [README](../README.md) - 完整文档
- [快速入门](QUICKSTART.md) - 快速上手指南
- [数据集结构](DATASET_STRUCTURE.md) - 数据组织说明
- [元数据 Schema](clip_metadata_schema.json) - JSON 格式定义

## 🆘 需要帮助？

如果遇到问题，请查看：
1. [故障排除部分](../README.md#-故障排除)
2. 运行测试脚本：`uv run python tests/manual_tests/test_input_adapter.py`
3. 检查日志文件：`logs/auto_annotator.log`
