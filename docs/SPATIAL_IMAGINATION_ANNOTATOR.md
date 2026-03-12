# Spatial Imagination Annotator

`scripts/annotate_spatial_imagination.py` is a dedicated batch annotator for the `Spatial_Imagination` task.

## What it uses from AutoAnnotator config

The script reads `config/config.yaml` through `auto_annotator.get_config()` and reuses:
- Gemini backend/model/generation settings (`gemini.*`).
- Default concurrency from `batch_processing.num_workers`.

It does **not** require a separate task config file.

## Input

- Dataset layout: `data/Dataset/{sport}/{event}/clips/{id}.json` + `{id}.mp4`.
- Only clips with `tasks_to_annotate` containing `Spatial_Imagination` are processed.
- The clip JSON must contain `source_annotation`; `tracking_bboxes` is ignored and never sent to model.

## Output

- Default output root: `data/output`.
- Output file: `data/output/{sport}/{event}/clips/{id}.json`.
- Annotation payload:
  - `task_L2: "Spatial_Imagination"`
  - `question: str`
  - `answer: str`

## Behavior

- Uses full clip frames; no frame-window output.
- Sends each clip directly as Gemini `inline_data`; it does not stage videos through GCS or the Gemini File API.
- Supports parallel processing (`--num-workers`) and tqdm progress bar (disable with `--no-progress`).
- Incremental by default: skips clips that already have a valid `Spatial_Imagination` annotation.
- Use `--overwrite` to re-run existing ones.

## Example

```bash
uv run python scripts/annotate_spatial_imagination.py \
  --dataset-root data/Dataset \
  --output-root data/output \
  --prompt-path config/prompts/spatial_imagination.md
```
