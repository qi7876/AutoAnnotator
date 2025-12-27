# AutoAnnotator - 智能视频标注系统

基于 Google Gemini API 的多模态体育视频自动标注系统。

## 📋 项目简介

AutoAnnotator 是一个专为多模态体育数据集设计的 Python 标注工具包。它利用 Google 的 Gemini API 执行 7 种不同的标注任务，覆盖感知和理解两个层次。

### 核心特性

- **7 种标注任务**：
  - 计分板理解（单帧）
  - 计分板理解（多帧）
  - 物体空间关系识别
  - 时空定位
  - 连续动作描述
  - 连续事件描述
  - 物体跟踪

- **模块化架构**：
  - 解耦的输入适配器，支持灵活的数据格式
  - 任务专用的标注器，统一接口设计
  - 可扩展的边界框和跟踪接口
  - 完善的配置管理系统

- **生产就绪**：
  - 健壮的错误处理和日志记录
  - JSON 验证和合并工具
  - 批量处理支持
  - 临时输出管理

## 🚀 快速开始

### 环境要求

- Python 3.10 或更高版本
- [uv](https://github.com/astral-sh/uv) 包管理器
- Google AI Studio API 密钥

### 安装步骤

1. **克隆仓库**：
```bash
git clone <repository-url>
cd AutoAnnotator
```

2. **安装依赖**：
```bash
uv sync
```

3. **配置 API 密钥**：
```bash
cp config/.env.example config/.env
# 编辑 config/.env，添加你的 GEMINI_API_KEY
```

4. **验证安装**：
```bash
uv run python -c "from auto_annotator import get_config; print('安装成功！')"
```

## ⚙️ 配置说明

### 环境变量

创建 `config/.env` 文件：

```env
GEMINI_API_KEY=your_api_key_here
PROJECT_ROOT=/path/to/AutoAnnotator
DATASET_ROOT=/path/to/AutoAnnotator/data/Dataset
```

### 配置文件

编辑 `config/config.yaml` 以自定义：

- Gemini 模型设置
- 输出目录
- 任务特定参数
- 日志配置

## 📊 数据集结构

### 目录组织

```
data/Dataset/
└── {运动项目}/              # 如：Archery, 3x3_Basketball
    └── {比赛事件}/          # 如：Men's_Individual, Men
        ├── {video_id}.mp4      # 原始视频文件（1.mp4, 2.mp4, ...）
        ├── {video_id}.json     # 原始视频元数据
        ├── metainfo.json       # 事件级元信息
        ├── clips/              # 视频片段目录
        │   ├── {id}.mp4
        │   └── {id}.json
        └── frames/             # 单帧图片目录
            ├── {id}.jpg
            └── {id}.json
```

### 元数据格式

**视频片段**（`clips/`）：

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
  "tasks_to_annotate": ["UCE", "Continuous_Actions_Caption"]
}
```

**单帧图片**（`frames/`）：

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
  "tasks_to_annotate": ["ScoreboardSingle"]
}
```

**关键特性**：
- 片段和单帧使用**统一的元数据格式**
- 通过 `total_frames` 区分类型（1 = 单帧，>1 = 片段）
- 简化的 ID 格式，与文件名保持一致

详细说明请参考：[docs/DATASET_STRUCTURE.md](docs/DATASET_STRUCTURE.md)

## 💻 使用方法

### 批量处理脚本

使用批处理脚本遍历数据集：
```bash
uv run python scripts/batch_processing.py
```

### 数据集标注监测脚本

用于统计每个原始视频元数据 JSON 中的 13 种任务数量，并可按 Source 写入飞书表格 B-N 列。

1. 配置 `config/.env`：
```env
APP_ID=your_app_id
APP_SECRET=your_app_secret
SPREADSHEET_TOKEN=your_spreadsheet_token
SHEET_ID=your_sheet_id
SOURCE=1
```

2. 本地统计（不上传）：
```bash
uv run scripts/monitor.py
```

3. 统计并上传：
```bash
uv run scripts/monitor.py --upload
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
from auto_annotator.config import get_config

# 获取配置
config = get_config()

# 初始化组件
gemini_client = GeminiClient()
prompt_loader = PromptLoader()
bbox_annotator = BBoxAnnotator(gemini_client)
tracker = ObjectTracker()

# 加载片段元数据
clip_metadata = InputAdapter.load_from_json(
    Path("data/Dataset/Archery/Men's_Individual/frames/1.json")
)

# 判断类型
if clip_metadata.info.is_single_frame():
    print("这是单帧图片")
elif clip_metadata.info.is_clip():
    print("这是视频片段")

# 创建特定任务的标注器
annotator = TaskAnnotatorFactory.create_annotator(
    task_name="ScoreboardSingle",
    gemini_client=gemini_client,
    prompt_loader=prompt_loader,
    bbox_annotator=bbox_annotator,
    tracker=tracker
)

# 执行标注
annotation = annotator.annotate(
    clip_metadata,
    dataset_root=config.dataset_root
)
print(annotation)

# 获取路径信息
video_path = clip_metadata.get_video_path(config.dataset_root)
original_video = clip_metadata.get_original_video_path(config.dataset_root)
```

## 📤 输出格式

标注结果保存为 JSON 文件：

```json
{
  "id": "1",
  "origin": {
    "sport": "3x3_Basketball",
    "event": "Men"
  },
  "annotations": [
    {
      "annotation_id": "1",
      "task_L1": "Understanding",
      "task_L2": "ScoreboardSingle",
      "timestamp_frame": 50,
      "question": "根据计分板，谁排在第一位？",
      "answer": "湖人队排在第一位。",
      "bounding_box": [934, 452, 1041, 667]
    }
  ]
}
```

## 🏗️ 项目结构

```
AutoAnnotator/
├── config/
│   ├── prompts/              # 任务提示词模板
│   ├── config.yaml           # 主配置文件
│   └── .env.example          # 环境变量模板
├── src/auto_annotator/
│   ├── adapters/             # 输入格式适配器
│   │   └── input_adapter.py # 统一的元数据处理
│   ├── annotators/           # 任务标注器和 AI 客户端
│   │   ├── base_annotator.py      # 标注器基类
│   │   ├── task_annotators.py     # 7 个任务标注器
│   │   ├── gemini_client.py       # Gemini API 客户端
│   │   ├── bbox_annotator.py      # 边界框标注接口
│   │   └── tracker.py             # 物体跟踪接口
│   ├── utils/                # 工具模块
│   │   ├── prompt_loader.py       # 提示词加载器
│   │   ├── video_utils.py         # 视频处理工具
│   │   └── json_utils.py          # JSON 工具
│   ├── config.py             # 配置管理
│   └── main.py               # 主入口
├── tests/                    # 单元测试
│   ├── manual_tests/          # 手动测试脚本
│   │   ├── test_input_adapter.py
│   │   ├── test_object_tracking_real_fixed.py
│   │   └── test_scoreboard_single_real.py
├── examples/                 # 示例文件
│   ├── example_segment_metadata.json
│   └── example_singleframe_metadata.json
├── docs/                     # 文档
│   ├── DATASET_STRUCTURE.md  # 数据集结构说明
│   ├── clip_metadata_schema.json  # 元数据 Schema
│   └── MIGRATION_GUIDE.md    # 迁移指南
├── tests/                    # 单元测试
└── data/output/
    ├── temp/                 # 临时标注输出
    └── final/                # 最终合并标注
```

## 🔧 开发指南

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_config.py

# 运行覆盖率测试
uv run pytest --cov=auto_annotator

# 测试适配器
uv run python tests/manual_tests/test_input_adapter.py
```

### 添加新任务

1. 在 `config/prompts/` 中创建提示词模板
2. 在 `task_annotators.py` 中实现标注器类，继承自 `BaseAnnotator`
3. 在 `TaskAnnotatorFactory` 中注册新任务
4. 更新配置和文档

示例：

```python
class NewTaskAnnotator(BaseAnnotator):
    """新任务标注器"""

    def get_task_name(self) -> str:
        return "NewTask"

    def get_task_l1(self) -> str:
        return "Understanding"  # 或 "Perception"

    def annotate(
        self,
        segment_metadata: ClipMetadata,
        dataset_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        # 实现标注逻辑
        video_path = segment_metadata.get_video_path(dataset_root)
        # ... 调用 Gemini API
        return annotation_result
```

## 🐛 故障排除

### API 密钥问题

如果看到 "GEMINI_API_KEY not found" 错误：
1. 确保 `config/.env` 文件存在
2. 检查 API 密钥是否正确设置
3. 在 Google AI Studio 中验证密钥有效性

### 视频上传超时

如果视频上传超时：
1. 增加 `config/config.yaml` 中的 `upload_timeout_sec`
2. 检查网络连接
3. 考虑将大视频分割成小片段

### 导入错误

如果遇到导入错误：
```bash
# 重新安装依赖
uv sync --reinstall

# 检查 Python 版本
python --version  # 应该是 3.10+
```

### 元数据验证失败

如果元数据验证失败：
1. 检查 JSON 格式是否正确
2. 确保 `total_frames` 和 `duration_sec` 一致
3. 验证文件路径是否存在
4. 参考 [docs/clip_metadata_schema.json](docs/clip_metadata_schema.json)

## 🔄 工作流集成

AutoAnnotator 设计为 5 步标注流程中的第 3 步：

1. **片段切分**：从完整视频中提取相关片段
2. **人工审核**：验证片段质量
3. **AI 标注**（AutoAnnotator）：生成标注
4. **人工审核**：验证和修正标注
5. **JSON 合并**：合并为最终数据集

## 📚 相关文档

- [数据集结构说明](docs/DATASET_STRUCTURE.md) - 详细的目录和元数据格式说明
- [元数据 Schema](docs/clip_metadata_schema.json) - 完整的 JSON Schema 定义
- [快速开始指南](docs/QUICKSTART.md) - 快速入门教程（待补充）
- [使用示例](docs/USAGE_EXAMPLES.md) - 更多使用示例（待补充）


## 📄 许可证

MIT License
