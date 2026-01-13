# 标注审查指南（文件浏览器版）

本指南适用于人工审查以下任务的标注结果，并在文件层面完成修改与保存：

- ScoreboardSingle
- ScoreboardMultiple
- Spatial_Temporal_Grounding
- Continuous_Actions_Caption
- Continuous_Events_Caption

## 目录约定（data 目录）

原始数据集：

- Clip 视频：`data/Dataset/{sport}/{event}/clips/{clip_id}.mp4`
- Frame 单帧：`data/Dataset/{sport}/{event}/frames/{frame_id}.jpg`

AI 标注结果：

- Clip 标注：`data/output/{sport}/{event}/clips/{clip_id}.json`
- Frame 标注：`data/output/{sport}/{event}/frames/{frame_id}.json`

## 审查流程（通用）

1. 在 `data/output/.../*.json` 中打开标注文件。
2. 找到 `annotations` 数组，按 `task_L2` 筛选需要审查的任务。
3. 根据 JSON 中的 `origin` 字段定位原始数据：
   - `origin.sport` / `origin.event` / `id`
4. 在 `data/Dataset/...` 打开对应 clip 或 frame，对照检查并修改 JSON。
5. 审查完成后，设置 `reviewed: true`。

## 任务与数据类型对应关系

- ScoreboardSingle：**Frame** 任务  
  标注 JSON 在 `data/output/.../frames/{frame_id}.json`  
  对照图片在 `data/Dataset/.../frames/{frame_id}.jpg`

- ScoreboardMultiple：**Clip** 任务  
  标注 JSON 在 `data/output/.../clips/{clip_id}.json`  
  对照视频在 `data/Dataset/.../clips/{clip_id}.mp4`

- Spatial_Temporal_Grounding：**Clip** 任务  
  标注 JSON 在 `data/output/.../clips/{clip_id}.json`  
  对照视频在 `data/Dataset/.../clips/{clip_id}.mp4`

- Continuous_Actions_Caption：**Clip** 任务  
  标注 JSON 在 `data/output/.../clips/{clip_id}.json`  
  对照视频在 `data/Dataset/.../clips/{clip_id}.mp4`

- Continuous_Events_Caption：**Clip** 任务  
  标注 JSON 在 `data/output/.../clips/{clip_id}.json`  
  对照视频在 `data/Dataset/.../clips/{clip_id}.mp4`

## 查找示例

标注 JSON：

```
data/output/Archery/Men's_Individual/clips/1.json
```

对应视频：

```
data/Dataset/Archery/Men's_Individual/clips/1.mp4
```

## reviewed 字段

每条标注建议包含：

```
"reviewed": true
```

若未审查请保持 `false` 或不填。

## 常见检查点

- `task_L2` 是否匹配目标任务
- `question` / `answer` 是否正确、清晰
- `timestamp_frame` / `Q_window_frame` / `A_window_frame` 是否在合理范围
- 对于 Spatial_Temporal_Grounding / Continuous_Actions_Caption，检查 `first_bounding_box` 与 `tracking_bboxes` 是否合理

