You are an expert video data annotator.

I will provide you with a video segment (10 FPS). You need to annotate a task called "Scoreboard Understanding - Multiple Frames". The scoreboard content changes between the beginning and end of the video. Your task is to create a question about these changes and identify the time windows where relevant scoreboard information appears.

Please annotate according to the following JSON template:

```json
{{
    "annotation_id": "1",
    "task_L1": "Understanding",
    "task_L2": "ScoreboardMultiple",
    "Q_window_frame": [10, 20],
    "question": "Based on the scoreboard information, describe the ranking changes between the second round and the first round.",
    "A_window_frame": ["10-13", "17-20"],
    "answer": "According to the scoreboard, France moved from second place to first place, England moved from first place to second place, and the rankings of other countries remained unchanged."
}}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will maintain these fields, you don't need to annotate them
- `Q_window_frame`: The time window (in original video frame numbers) for the question, covering the entire segment where changes occur
- `question`: Create a question about the changes in scoreboard content (e.g., ranking changes, score progression)
- `A_window_frame`: Time windows (in original video frame numbers) showing the relevant scoreboard states referenced in the answer. Provide exactly two windows showing the before and after states.
- `answer`: Provide a detailed answer describing the changes between the two scoreboard states

The segment's starting frame in the original video is frame {num_first_frame}.
The segment has {total_frames} frames in total.

Please analyze the video content and return your annotation in JSON format.
