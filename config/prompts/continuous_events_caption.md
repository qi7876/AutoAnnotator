You are an expert video data annotator.

I will provide you with a video segment (10 FPS, you need to convert seconds to frame numbers). You need to annotate a task called "Continuous Events Caption". Your task is to continuously describe the events occurring in the video and the corresponding time periods.

Please annotate according to the following JSON template:

```json
{{
    "annotation_id": "1",
    "task_L1": "Understanding",
    "task_L2": "Continuous_Events_Caption",
    "Q_window_frame": [15, 20],
    "question": "Please continuously describe the events occurring in the video, and output the corresponding time intervals.",
    "A_window_frame": ["15-18", "19-20"],
    "answer": [
        "The offensive player successfully breaks through",
        "The offensive player successfully scores"
    ]
}}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will maintain these fields, you don't need to annotate them
- `Q_window_frame`: The question frame window, using 0-indexed clip frame numbers. Valid range: 0..{max_frame} (inclusive). Ensure start <= end.
- `question`: Create a question asking for continuous event description
- `A_window_frame`: Answer frame windows, using 0-indexed clip frame numbers. Each segment must be within 0..{max_frame} (inclusive). The time segments should be continuous and non-overlapping.
- `answer`: The answers should align with the number of answer frame windows. Each answer describes a high-level event (not individual actions, but meaningful events like "successful breakthrough", "goal scored", "defensive steal")
 
The segment has {total_frames} frames in total.
Valid clip frame indices: 0..{max_frame} (inclusive).

Please note the response format. The time periods in A_window_frame are segmented, and the number of segments should align with the number of answers in the answer array.

Please note:
- Events are higher-level than actions (e.g., "successful score" vs "jumps and shoots")
- Events describe outcomes and significant moments, not detailed movement descriptions
- Events may involve multiple players or the game state

Please analyze the video content, combine it with the JSON template format, and return your annotation results.
