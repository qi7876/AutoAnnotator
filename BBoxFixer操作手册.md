# BBoxFixer 使用指南（含 flagged 模式）

BBoxFixer 是 AutoAnnotator 内置的 GUI 工具，用来人工修正/补全 MOTChallenge 格式的跟踪框（`*.txt`），并在需要时把 JSON 里的部分字段（如 `reviewed`、`first_bounding_box`）与 MOT 文件同步。

> 适用场景
> - 标注员/复查员：逐帧检查、拖拽修正框、必要时新增/删除框
> - 质检/返工：优先处理 `retrack=true` 或 `is_window_consistence=false` 的任务（flagged 模式）

---

## 1. 安装与启动

在 AutoAnnotator 目录下：

```bash
uv sync

# 常规模式：加载所有能找到 mot_file 的任务
uv run python scripts/bbox_fixer_cli.py

# flagged 模式：只加载 retrack 或 is_window_consistence=false 的任务
uv run python scripts/bbox_fixer_cli.py --mode flagged
```

### 默认数据路径
入口脚本 [scripts/bbox_fixer_cli.py](scripts/bbox_fixer_cli.py) 固定使用：

- 数据集根目录：`data/Dataset`
- 输出根目录：`data/output`
- 状态文件：`data/bbox_fixer_state.json`

如需改路径，直接编辑该脚本中的 `dataset_root` / `output_root` / `state_path`。

---

## 2. 数据发现规则（它到底会加载哪些任务）

BBoxFixer 的任务发现逻辑（简述）如下：

1. 遍历 `data/output/{sport}/{event}/clips/*.json`（以输出 JSON 为准；标记字段来自这里）。
2. 对每个 `{clip_id}.json`：在 `annotations[]` 中找到带有 `tracking_bboxes.mot_file` 的 annotation。
     - 每个 annotation 会形成一个“任务”（task），其显示名通常来自 `task_L2`。
3. 根据 `{sport}/{event}/{clip_id}` 去定位视频文件：
     - `data/Dataset/{sport}/{event}/clips/{clip_id}.mp4`
     - 若视频不存在，BBoxFixer 会跳过该任务（否则 GUI 无法展示画面）。

### flagged 模式的筛选规则
仅当 annotation 满足以下任意条件才会被加载：

- `retrack == true`，或
- `is_window_consistence == false`

并且同时需要 `tracking_bboxes.mot_file` 存在。

---

## 3. 关键约定：帧号、window 语义、文件路径

### 3.1 帧号（非常重要）

- **MOT TXT（MOTChallenge）帧号：1-based**
    - 第 1 帧写作 `frame=1`
- **JSON 的 window（如 `Q_window_frame`/`A_window_frame`）：0-based**
    - 第 1 帧在 JSON window 中对应 `0`

BBoxFixer 界面中显示/跳转的帧号与 MOT 存储一致，通常以 **1-based** 进行操作；而当它读取 JSON window 决定“窗口起始帧”时，会做 `0 -> 1` 的转换。

### 3.2 window 都表示“区间”（闭区间）

BBoxFixer 当前把 window 统一当作闭区间处理：

- `A_window_frame = [start, end]`：表示从 `start` 到 `end` 的 **闭区间**（包含两端）。
- `A_window_frame = ["10-20", "30-35"]`：表示多个区间，BBoxFixer 会用最小 start 与最大 end 作为整体窗口范围（用于界面参考/保存窗口时取起始帧）。

> 说明：不同数据可能混用多种格式；BBoxFixer 会尽量解析常见写法。

### 3.3 `mot_file` 路径与“保存到了哪里”

`tracking_bboxes.mot_file` 可能是：

- 绝对路径（例如 `/mnt/.../xxx.txt`），或
- 相对路径（相对 AutoAnnotator 仓库根目录）。

BBoxFixer 会按 JSON 里的 `mot_file` 去读/写 MOT 文件：

- 如果你发现“界面保存了但文件没变”，首先确认你正在查看的是否就是 JSON 指向的那个 `mot_file`。

---

## 4. 界面操作

### 4.1 帧与 clip 切换

- 上一帧/下一帧：按钮或快捷键 `A` / `D`
- 跳转帧：在 Frame 输入框输入帧号，回车或点 `Go`
- 切换 clip：`<< Prev Clip` / `Next Clip >>`

### 4.2 编辑框（拖拽）

- 每个框有两个控制点（左上、右下）。
- 拖动控制点可以改变框的位置与大小。

### 4.3 新增框

- 点击 `新增框`：在当前帧生成一个默认大小的框（居中），并自动分配 `track_id`。
- 生成后可拖拽调整；保存时写入 MOT。

### 4.4 删除框

- 先用鼠标点击框，使其被选中（框会进入 selected 状态）。
- 点击 `删除框`，或按键盘 `Delete` / `Backspace` 删除。

如果未选中任何框，会在日志区提示。

### 4.5 缩放

- `Fit`：适配窗口
- `Zoom + / Zoom -`：缩放
- Ctrl + 鼠标滚轮：缩放

---

## 5. 保存行为（常规模式 vs flagged 模式）

### 5.1 自动保存（两种模式都适用）

BBoxFixer 在以下时机会自动把当前修改写回 MOT：

- 切换 clip 时
- 关闭窗口时

此外，`Reviewed` 复选框会写回当前 clip 的 JSON 对应 annotation：

- `annotations[ann_index].reviewed = true/false`

### 5.2 flagged 模式专用：保存窗口（`保存窗口` 按钮）

在 flagged 模式下会多一个 `保存窗口` 按钮。它的语义是：

1. 先保存当前 MOT（和常规保存一致）。
2. **不修改** JSON 里的窗口范围字段（不会改 `A_window_frame` / `answer_window`）。
3. 把 JSON 的 `first_bounding_box` 同步为“窗口起始帧”对应的 MOT 框坐标：
     - 窗口起始帧来自 JSON window（0-based），同步时转换为 MOT 的 1-based 帧号。
     - 若起始帧有多个框：
         - 如果 JSON 里已有 `first_bounding_box`，会选取与其中心点最接近的框；
         - 否则默认选 `track_id` 最小的框。

当窗口起始帧没有任何框时，BBoxFixer 会提示并跳过更新 `first_bounding_box`。

---

## 6. 界面参考信息（Annotation 参考面板）

界面中会显示当前 annotation 的关键信息，便于复查：

- `Q_window_frame`
- `A_window_frame`
- `answer_window`（若存在）
- `answer` 文本
- `MOT 帧区间(有框)`：当前 MOT 文件里“哪些帧存在框”的区间合并结果
    - **该显示为 0-based**（即把 MOT 的 1-based 帧号整体减 1 后展示）

---

## 7. 常见问题（FAQ）

### Q1：启动时报 `No clips found in dataset.`
通常是以下原因之一：

- `data/output/{sport}/{event}/clips/*.json` 不存在或目录结构不匹配
- JSON 的 `annotations[]` 里没有 `tracking_bboxes.mot_file`
- 对应视频 `data/Dataset/{sport}/{event}/clips/{clip_id}.mp4` 不存在（目前会被跳过）

### Q2：我改了框，保存后 MOT 文件没变化？
优先检查：

- 你打开查看的 MOT 文件是否就是 JSON 的 `tracking_bboxes.mot_file` 指向的文件（可能是绝对路径）。

### Q3：`retrack` 按钮不可用
BBoxFixer 的重跟踪依赖可选组件 `ObjectTracker`；如果运行环境里没有正确安装/配置对应 tracker（例如 SAM2 权重），按钮会被禁用。

---

## 8. 开发者入口（可选）

- 入口脚本： [scripts/bbox_fixer_cli.py](scripts/bbox_fixer_cli.py)
- 主界面实现： [src/bbox_fixer/viewer.py](src/bbox_fixer/viewer.py)
- MOT 读写： [src/bbox_fixer/mot_io.py](src/bbox_fixer/mot_io.py)