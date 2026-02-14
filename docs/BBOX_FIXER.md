# BBoxFixer

该工具用于人工纠正 MOT 跟踪框，面向众包标注员，强调简单易用与自动保存。

## 依赖

- Python 3.12+
- PySide6
- opencv-python

安装依赖：

```bash
uv sync
```

## 启动

```bash
uv run python scripts/bbox_fixer_cli.py
```

## 默认读取路径

当前路径在 `scripts/bbox_fixer_cli.py` 中固定为：

- 数据集：`data/Dataset`
- MOT 输出：`data/output`
- 状态保存：`data/bbox_fixer_state.json`

如果你需要修改路径，请直接编辑 `scripts/bbox_fixer_cli.py` 中的 `dataset_root` 与 `output_root`。

## 数据与文件格式

- 视频：`data/Dataset/{sport}/{event}/clips/{clip_id}.mp4`
- MOT：`data/output/{sport}/{event}/clips/mot/{clip_id}_{task_name}.txt`
- MOTChallenge 2D 格式：
  ```
  <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
  ```
  其中 frame/id/box 为 1-based，conf/x/y/z 默认 -1。

## 操作方式

- **切换帧**：左右方向键，或下方按钮
- **跳转帧**：输入框输入帧号 + 回车/Go
- **切换 clip**：左右两侧按钮
- **编辑框**：拖动左上角/右下角控制点
- **缩放**：Fit/Zoom+/Zoom- 按钮
- **重新跟踪**：点击 “从此帧重跟踪” 将使用 SAM2 跟踪器从当前帧一路自动生成到任务窗口结束帧的 MOT 框；需要提前准备好 `auto_annotator/annotators/sam2` 权重
- **保存时机**：
  - 切换 clip 时自动保存
  - 关闭窗口时自动保存

## 状态恢复

退出时会记录当前 clip 与帧号到：

```
data/bbox_fixer_state.json
```

下次启动自动恢复。

## 提示区

帧画面下方有文本提示区：

- 当前 clip/帧
- 保存状态
- 本帧无框提示（如 “No boxes for frame X”）

## 注意事项

- 工具只读取已有标注里的 mot_file/tracking_bboxes
- 命名规则：`{clip_id}_{task_name}.txt`
- MOT 写入为原子写：先写入 `.tmp` 再覆盖
