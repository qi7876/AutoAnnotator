# AutoAnnotator 7种任务标注流程规范

本文档规范记录 AutoAnnotator 支持的 7 种任务的标注流程与关键字段。

## 共通流程

1. 读取片段元数据，定位视频文件。
2. 上传视频到 Gemini。
3. 加载对应任务的提示词模板（config/prompts/*.md）。
4. 调用 Gemini 返回结构化 JSON。
5. 视任务进行后处理（抽帧、bbox、跟踪、校验）。
6. 补齐通用元数据字段并返回结果。
7. 清理上传文件。

## 任务流程

### 1) ScoreboardSingle（单帧记分板理解）

- 目标：在视频中找到完整清晰的记分板帧，并生成问答。
- 输出字段：
  - timestamp_frame, question, answer, bounding_box(自然语言描述)
- 后处理：
  - 抽取 timestamp_frame 对应帧
  - 使用 BBoxAnnotator 将 bounding_box 描述转成坐标
  - 写回 bounding_box 坐标数组

### 2) ScoreboardMultiple（多帧记分板变化）

- 目标：描述记分板在两个时间点的变化。
- 输出字段：
  - Q_window_frame, A_window_frame(两个窗口), question, answer
- 后处理：
  - 无额外后处理，仅补齐通用元数据

### 3) Objects_Spatial_Relationships（物体空间关系）

- 目标：在单帧内描述多个对象的空间关系。
- 输出字段：
  - timestamp_frame, question, answer, bounding_box(对象描述列表)
- 后处理：
  - 当前实现保留自然语言描述（bbox 坐标生成未实现）

### 4) Spatial_Temporal_Grounding（时空定位）

- 目标：对特定对象的动作进行精确时空定位描述。
- 输出字段：
  - question, A_window_frame, first_frame_description
- 后处理：
  - 抽取 A_window_frame 首帧
  - bbox + 跟踪逻辑当前未实现，仅保留描述

### 5) Continuous_Actions_Caption（连续动作描述）

- 目标：连续描述一个运动员的动作序列。
- 输出字段：
  - Q_window_frame, A_window_frame(多个片段), question, answer(动作列表)
- 后处理：
  - 校验 A_window_frame 数量与 answer 数量一致（不一致仅告警）

### 6) Continuous_Events_Caption（连续事件描述）

- 目标：连续描述更高层级事件（如进球、突破成功）。
- 输出字段：
  - Q_window_frame, A_window_frame(多个片段), question, answer(事件列表)
- 后处理：
  - 校验 A_window_frame 数量与 answer 数量一致（不一致仅告警）

### 7) Object_Tracking（目标跟踪）

- 目标：对单个对象进行时间窗内跟踪。
- 输出字段：
  - Q_window_frame, query, first_frame_description(描述文本)
- 后处理：
  - 抽取 Q_window_frame 首帧
  - 用 BBoxAnnotator 生成首帧 bbox
  - 用 ObjectTracker 生成 tracking_bboxes

## 参考实现

- 任务实现：src/auto_annotator/annotators/task_annotators.py
- 提示词模板：config/prompts/*.md
