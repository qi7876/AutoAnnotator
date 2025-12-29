You are an expert video data annotator.

I will provide you with a video segment (10 FPS). You need to annotate a task called "Spatial-Temporal Grounding". Your task is to select an object in the video and provide a very detailed description of its behavior, state, and attributes.

Please annotate according to the following JSON template:

```json
{{
    "annotation_id": "1",
    "task_L1": "Understanding",
    "task_L2": "Spatial_Temporal_Grounding",
    "question": "The player wearing red jersey number 12 from the Spanish team completes the second shot",
    "A_window_frame": [15, 18],
    "first_frame_description": "the player in red jersey number 12 on the left side of the court"
}}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will maintain these fields, you don't need to annotate them
- `question`: Provide a very detailed description of a specific object/player and their action. Include:
  - Detailed appearance description (team, jersey color, number, physical characteristics)
  - Specific action or event
  - Temporal qualifiers (e.g., "second shot", "third attempt")
- `A_window_frame`: The time window (in clip frame numbers) when this specific action occurs
- `first_frame_description`: Natural language description of the object's location in the first frame of the answer window (e.g., "the player in red jersey number 12 on the left side of the court"). We will use this to generate the bounding box.

The segment has {total_frames} frames in total.

Please analyze the video content, select one prominent action or event, and return your annotation in JSON format. Make the description highly specific so it uniquely identifies the target object and action.
