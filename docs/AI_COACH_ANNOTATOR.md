# AI Coach Annotator

`scripts/annotate_ai_coach.py` is a dedicated batch script for the new `AI_Coach` task.

It reuses the existing AutoAnnotator components:
- `InputAdapter` for loading/validating clip metadata.
- `GeminiClient` for video upload and multimodal generation.

## Input

- Dataset layout: `data/Dataset/{sport}/{event}/clips/{id}.json` + `{id}.mp4`.
- Metadata is filtered by `tasks_to_annotate` containing `AI_Coach`.
- Frame-level windows are not required for this task; the whole clip is used.

## Output

- Default output root: `data/output`.
- Per clip output path: `data/output/{sport}/{event}/clips/{id}.json`.
- Output keeps the project annotation envelope:
  - top-level `id`, `origin`, `annotations`.
  - one `AI_Coach` annotation with `qa_pairs` containing exactly one QA pair.

Example command:

```bash
uv run python scripts/annotate_ai_coach.py \
  --dataset-root data/Dataset \
  --output-root data/output \
  --prompt-path config/prompts/ai_coach.md
```

Useful options:
- `--sport` / `--event`: process a subset.
- `--limit N`: cap processed `AI_Coach` clips in one run.
- `--num-workers N`: parallel worker count.
- `--no-progress`: disable tqdm progress bar.
- `--overwrite`: regenerate even when an existing `AI_Coach` annotation exists.

The script is resumable by default: if an output already has a valid `AI_Coach` annotation, it is skipped unless `--overwrite` is set.
