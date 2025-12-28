You are an expert video data annotator.

I will provide you with a video segment (10 FPS). You need to annotate a task called "Object Tracking". Your task is to describe a single target to be tracked.

Please return JSON in the following format:

```json
{{
  "annotation_id": "1",
  "task_L1": "Perception",
  "task_L2": "Object_Tracking",
  "Q_window_frame": [10, 20],
  "query": "Track the player in red jersey number 10 during the play.",
  "first_frame_description": "the player in red jersey number 10 on the right side of the court"
}}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will fill these fields.
- `Q_window_frame`: Frame window for tracking (original video frame numbers, cover the full segment).
- `query`: One sentence describing the tracking target.
- `first_frame_description`: One concise description of the target in the first frame. Include appearance and location.

The segment starts at frame {num_first_frame} in the original video and has {total_frames} frames in total.

Return JSON only.
