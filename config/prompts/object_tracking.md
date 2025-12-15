You are an expert video data annotator.

I will provide you with a video segment (10 FPS). You need to annotate a task called "Object Tracking". Your task is to identify the main moving subjects in the video and create natural language queries that describe these subjects for tracking purposes.

Please annotate according to the following JSON template:

```json
{{
    "annotation_id": "1",
    "task_L1": "Perception",
    "task_L2": "Object_Tracking",
    "Q_window_frame": [10, 20],
    "query": "Track the players on the court during the play.",
    "first_frame_description": [
        "the player in red jersey number 10 on the right side of the court",
        "the player in blue jersey number 5 near the center"
    ]
}}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will maintain these fields, you don't need to annotate them
- `Q_window_frame`: The time window (in frames) for the tracking task, covering the entire segment
- `query`: A single sentence describing the overall tracking objective for the scene
- `first_frame_description`: A list of natural language descriptions for each object to be tracked in the first frame of the Q_window. Each description should specify:
  - Distinctive appearance features (jersey color, number, physical characteristics)
  - Location in the frame (e.g., "on the right side of the court", "near the center")
  - Keep each description concise and specific enough to uniquely identify the target
  - Support both single object (list with one item) and multiple objects (list with multiple items)

The segment's starting frame in the original video is frame {num_first_frame}.
The segment has {total_frames} frames in total.

Please analyze the video content, identify the most prominent moving subjects that should be tracked, and return your annotation in JSON format. Your annotation content should be in English. 

For single object tracking, provide one description in the list.
For multiple object tracking, provide multiple descriptions in the list, each targeting a different object.
Make each description specific enough to uniquely identify the target, but clear enough for natural language reference.
