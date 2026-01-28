# Video Captioner

该工具用于处理 `caption_data/Dataset/{sport}/{event}/1.mp4`，自动生成视频 Caption（短片段 + 长片段密集解说）。

## 工作流程

1. **抽取长片段（关键帧裁剪，不重新编码）**
   - 若原视频时长 < 5 分钟：直接使用全视频。
   - 否则：目标片段时长随原视频变长而变长（默认 `0.8 * 原时长`），并限制在 `[5min, 30min]`。
   - 片段在原视频中的起点随机。
2. **长片段再切分为 ~1 分钟短片段**（同样使用 `ffmpeg -c copy` 的关键帧裁剪机制）。
3. **对每个短片段生成密集结构化 Caption**（JSON：帧区间 + Caption；约 8–18 条/分钟），同时输出该短片段的 `chunk_summary`。
4. **连续性增强**：生成第 N 个短片段时，将第 N-1 个短片段的 `chunk_summary` 一起输入模型，以提高解说连贯性。
5. **长片段密集 Caption**：程序将所有短片段的 `spans` 通过帧号平移映射回原视频，拼接为长片段的密集解说（不做“有损合并”）。

## 增量恢复（Resume）

- 运行过程中会**增量写入** `chunk_captions.json`（每完成一个 chunk 就落盘一次），因此进程中断后可直接重跑继续。
- 默认（不加 `--overwrite`）行为：
  - 若 `segment.mp4` / `chunks/chunk_*.mp4` 已存在：直接复用，不重新裁剪。
  - 若 `chunk_captions.json` 中已有某些 `chunk_index`：跳过这些 chunk，继续生成剩余部分。
  - 最终会补齐 `long_caption.json` 与 `run_meta.json`。
- 若你希望完全重新生成该 sport/event 的所有结果：使用 `--overwrite`。

## 运行方式

```bash
uv run python scripts/generate_captions.py --config video_captioner_config.toml
```

主要参数通过 `video_captioner_config.toml` 传递（默认模板在仓库根目录）。常用项：
- `run.model = "fake"`：离线调试（不调用 Gemini）
- `run.language = "en"`：输出英文（默认就是英文）
- `run.sport/run.event`：只处理某个 sport/event
- `run.max_events`：只跑前 N 个事件做验证
- `run.seed`：固定随机裁剪结果，便于复现
- `run.overwrite = true` 或命令行 `--overwrite`：覆盖已生成结果
- `run.progress = true/false`：控制 tqdm 进度条（省略则仅在 TTY 下显示）
- `logging.file`：指定日志文件（相对路径会拼到 `output_root/` 下）

## 输出目录结构

默认输出到 `caption_outputs/{sport}/{event}/`：
- `segment.mp4`：抽取的长片段
- `chunks/chunk_XXX.mp4`：按约 60 秒切分的短片段
- `chunk_captions.json`：每个短片段的结构化 Caption（含帧区间与原视频帧映射）
- `long_caption.json`：长片段密集解说（原视频帧号坐标系，含 `chunk_index`）
- `run_meta.json`：本次运行的裁剪参数与统计信息（含长片段原视频帧映射）

帧映射说明（与 AutoAnnotator 一致的字段命名）：
- 长片段：`run_meta.json` 的 `segment_info.original_starting_frame` + `segment_info.total_frames` + `segment_info.fps`
- 短片段：`chunk_captions.json` 每个条目的 `info.original_starting_frame` + `info.total_frames` + `info.fps`
- 将短片段内的 `spans[].start_frame/end_frame` 映射回原视频：  
  `source_frame = info.original_starting_frame + spans[].start_frame`（end 同理）

## 依赖与配置

- 系统依赖：`ffmpeg` / `ffprobe`
- 模型依赖：复用 AutoAnnotator 的 `GeminiClient`，需要在 `config/.env` 中配置 Gemini/Vertex 相关 Key 与（如适用）GCS bucket。
