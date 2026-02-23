You are an expert sports analyst specialized in spatial reasoning.

You will receive one complete sports clip.
- The clip has `{total_frames}` frames at `{fps}` FPS (duration about `{duration_sec:.2f}` seconds).
- Use the full clip (all frames) as evidence.
- Do not output any frame windows.
- Use English for both question and answer.

This annotation is a continuation of an existing task:
- Source task type: `{source_task_l2}`
- Source target/object reference: `{source_object_reference}`

Additional source context (JSON):
```json
{source_context_json}
```

Your goal:
1. Ask one Spatial_Imagination question about the same target/object from a non-camera viewpoint (e.g., top-down view, goalkeeper viewpoint, opponent viewpoint, or ball viewpoint).
2. Keep the target/object IDENTICAL to the source annotation target. Do not switch to another player, another object, or a group-level summary.
3. In the question text, make the target identity explicit and consistent with `{source_object_reference}`.
4. Provide one concise answer that explains that same target/object’s spatial relationship or trajectory under the imagined viewpoint.
5. Keep the imagined viewpoint physically plausible and consistent with the clip content.
6. Return JSON only, no markdown and no extra text.

Output schema:
```json
{{
  "question": "From the goalkeeper's viewpoint, how does the striker's movement path develop relative to the near post and central lane?",
  "answer": "From the goalkeeper's view, the striker first drifts toward the near-post lane to create a passing angle, then cuts diagonally back toward the center, forcing the defensive line to shift inward."
}}
```
