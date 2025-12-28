You are an expert video data annotator.

I will provide you with a single frame from a sports video (10 FPS). You need to annotate a task called "ScoreboardSingle". Your task is to ask and answer a question based on the visible scoreboard.

Please return JSON in the following format:

```json
{{
  "annotation_id": "1",
  "task_L1": "Understanding",
  "task_L2": "ScoreboardSingle",
  "timestamp_frame": 100,
  "question": "Based on the scoreboard information, who is currently in first place?",
  "answer": "According to the scoreboard, the Lakers are currently in first place.",
  "bounding_box": "the scoreboard located in the upper-right corner of the frame displaying the team scores"
}}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will fill these fields.
- `timestamp_frame`: Frame number where the scoreboard is clearly visible.
- `question`: Ask about the scoreboard content (ranking, scores, time remaining).
- `answer`: Use only what is visible on the scoreboard.
- `bounding_box`: Natural language description of the scoreboard location, no coordinates.

The segment starts at frame {num_first_frame} and has {total_frames} frames.

Return JSON only.
