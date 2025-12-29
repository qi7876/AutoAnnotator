# 快速入门指南

本指南将帮助你在 5 分钟内开始使用 AutoAnnotator。

## 🎯 目标

完成本指南后，你将能够：
- 正确安装和配置 AutoAnnotator
- 理解数据集结构
- 运行第一个标注任务
- 查看和理解输出结果

## 📋 准备工作

### 1. 系统要求

- Python 3.10 或更高版本
- 4GB+ 可用内存
- 网络连接（用于访问 Gemini API）

### 2. 获取 API 密钥

1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 点击 "Create API Key"
3. 复制生成的 API 密钥

## 🚀 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd AutoAnnotator
```

### 2. 安装 uv 包管理器

如果还没有安装 uv：

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. 安装依赖

```bash
uv sync
```

这会自动创建虚拟环境并安装所有依赖。

### 4. 配置环境

```bash
# 复制环境变量模板
cp config/.env.example config/.env

# 使用你喜欢的编辑器打开 .env 文件
nano config/.env  # 或 vim, code, 等
```

编辑 `config/.env`，填入你的配置：

```env
# Google Gemini API 后端与密钥
GEMINI_MODEL_API_KEY=your_model_api_key_here
GEMINI_GROUNDING_API_KEY=your_grounding_api_key_here

# 项目根目录（自动设置，通常不需要修改）
PROJECT_ROOT=/path/to/AutoAnnotator

# 数据集根目录
DATASET_ROOT=/path/to/AutoAnnotator/data/Dataset
```

### 5. 验证安装

```bash
uv run python -c "from auto_annotator import get_config; print('✓ 安装成功！')"
```

如果看到 "✓ 安装成功！"，说明安装完成。

## 📊 准备数据

### 数据集结构

AutoAnnotator 需要以下目录结构：

```
data/Dataset/
└── {Sport}/
    └── {Event}/
        ├── 1.mp4                    # 原始视频
        ├── 1.json                   # 原始视频元数据
        ├── segment_dir/             # 视频片段
        │   ├── 1_split_1_start_000652.mp4
        │   └── 1_split_1_start_000652.json
        └── singleframes_dir/        # 单帧图片
            ├── 5.jpg
            └── 5.json
```

### 使用示例数据

项目包含示例元数据文件：

```bash
# 查看片段示例
cat examples/example_segment_metadata.json

# 查看单帧示例
cat examples/example_singleframe_metadata.json
```

## 🎬 第一次运行

### 示例 1：测试单帧计分板标注

使用项目提供的测试脚本：

```bash
# 如果你有真实的数据集
uv run python tests/manual_tests/scoreboard_single_real.py \
    data/Dataset/Archery/Men\'s_Individual/singleframes_dir/5.json
```

### 示例 2：使用 Python API

创建测试文件 `test_annotation.py`：

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
from auto_annotator.config import get_config

# 获取配置
config = get_config()
print(f"数据集根目录: {config.dataset_root}")

# 初始化组件
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()

# 加载示例元数据
clip_metadata = InputAdapter.load_from_json(
    Path("examples/example_frame_metadata.json")
)

print(f"片段 ID: {clip_metadata.id}")
print(f"运动项目: {clip_metadata.origin.sport}")
print(f"比赛事件: {clip_metadata.origin.event}")
print(f"是否为单帧: {clip_metadata.info.is_single_frame()}")

# 验证元数据（不检查文件存在性）
is_valid, error = InputAdapter.validate_metadata(
    clip_metadata,
    check_file_existence=False
)

if is_valid:
    print("✓ 元数据验证通过")
else:
    print(f"✗ 元数据验证失败: {error}")

# 创建标注器
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="ScoreboardSingle",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

print(f"标注器任务: {annotator.get_task_name()}")
print(f"任务类别: {annotator.get_task_l1()}")

# 如果有真实视频文件，可以执行标注
# annotation = annotator.annotate(clip_metadata, config.dataset_root)
# print(annotation)
```

运行测试：

```bash
uv run python test_annotation.py
```

### 示例 3：批量处理

如果你有多个片段需要标注：

```bash
# 批量处理数据集
uv run python scripts/batch_processing.py
```

## 📤 理解输出

### 输出结构

标注结果保存在 `data/output/temp/` 目录：

```
data/output/
└── temp/
    └── Archery/
        └── Men's_Individual/
            └── annotation_5.json
```

### 输出格式

```json
{
  "id": "1",
  "origin": {
    "sport": "Archery",
    "event": "Men's_Individual"
  },
  "annotations": [
    {
      "annotation_id": "1",
      "task_L1": "Understanding",
      "task_L2": "ScoreboardSingle",
      "timestamp_frame": 0,
      "question": "根据计分板，当前的得分是多少？",
      "answer": "当前得分是 28 分。",
      "bounding_box": [100, 50, 300, 150]
    }
  ]
}
```

### 字段说明

- `id`: 片段唯一标识
- `origin`: 原始视频来源信息
- `annotations`: 标注列表
  - `annotation_id`: 标注唯一标识
  - `task_L1`: 任务一级分类（Perception/Understanding）
  - `task_L2`: 任务二级分类（具体任务名称）
  - `timestamp_frame`: 标注对应的帧号
  - `question`: 问题
  - `answer`: 答案
  - `bounding_box`: 边界框 [左上x, 左上y, 右下x, 右下y]（可选）

## 🔍 常见问题

### Q1: 如何检查 API 密钥是否正确？

```bash
uv run python -c "
from auto_annotator.config import get_config
config = get_config()
print(f'API 密钥已设置: {bool(config.gemini.api_key)}')
print(f'密钥长度: {len(config.gemini.api_key) if config.gemini.api_key else 0}')
"
```

### Q2: 元数据文件在哪里？

元数据文件与视频/图片文件在同一目录，扩展名为 `.json`：

- 片段: `segment_dir/1_split_1_start_000652.json`
- 单帧: `singleframes_dir/5.json`

### Q3: 如何验证元数据格式？

```bash
uv run python tests/manual_tests/test_input_adapter.py
```

或者：

```python
from pathlib import Path
from auto_annotator import InputAdapter

# 加载并验证
metadata = InputAdapter.load_from_json(Path("path/to/metadata.json"))
is_valid, error = InputAdapter.validate_metadata(
    metadata,
    check_file_existence=False
)
print(f"验证结果: {'通过' if is_valid else f'失败 - {error}'}")
```

### Q4: 支持哪些任务？

当前支持的任务：
- `ScoreboardSingle` - 单帧计分板理解
- `ScoreboardContinuous` - 多帧计分板理解
- `Objects_Spatial_Relationships` - 物体空间关系
- `Spatial_Temporal_Grounding` - 时空定位
- `Continuous_Actions_Caption` - 连续动作描述
- `Continuous_Events_Caption` - 连续事件描述
- `Object_Tracking` - 物体跟踪

### Q5: 如何只标注特定任务？

在元数据文件中指定 `tasks_to_annotate` 字段：

```json
{
  "id": 5,
  "origin": {...},
  "info": {...},
  "tasks_to_annotate": ["ScoreboardSingle"]
}
```

## 🐛 故障排除

### "GEMINI_MODEL_API_KEY not found"

确保创建了 `config/.env` 并设置了 API 密钥：

```bash
cp config/.env.example config/.env
# 编辑 config/.env 添加密钥
```
```
GEMINI_MODEL_API_KEY=your_model_api_key_here
GEMINI_GROUNDING_API_KEY=your_grounding_api_key_here
```

### "Video file not found"

检查：
1. 文件路径是否正确
2. 文件是否确实存在
3. 使用绝对路径或正确的相对路径

### 导入错误

重新安装依赖：

```bash
uv sync --reinstall

# 检查 Python 版本
python --version  # 应该是 3.10+
```

## 📚 下一步

现在你已经完成了基础设置，可以：

1. 阅读 [数据集结构说明](DATASET_STRUCTURE.md) 了解详细的数据组织
2. 查看 [使用示例](USAGE_EXAMPLES.md) 学习更多高级用法
3. 参考 [API 文档](../README.md#-python-api) 了解完整的 API
4. 自定义 [配置文件](../config/config.yaml)（包含 Vertex 视频的 GCS bucket 配置）
5. 自定义 [提示词模板](../config/prompts/)
