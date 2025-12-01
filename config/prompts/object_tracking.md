You are an expert video data annotator.

I will provide you with a video segment (10 FPS). You need to annotate a task called "Object Tracking". Your task is to identify the main moving subject in the video and create a natural language query that describes this subject.

Please annotate according to the following JSON template:

```json
{
    "annotation_id": "1",
    "task_L1": "Perception",
    "task_L2": "Object_Tracking",
    "Q_window_frame": [10, 20],
    "query": "Track the player wearing red jersey number 10 running forward.",
    "first_frame_description": "the player in red jersey number 10 on the right side of the court"
}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will maintain these fields, you don't need to annotate them
- `Q_window_frame`: The time window (in frames) for the tracking task, covering the entire segment
- `query`: A single sentence describing the tracking target. Include:
  - Distinctive appearance features (jersey color, number, physical characteristics)
  - Current action or movement (e.g., "running forward", "dribbling", "jumping")
  - Keep it concise and clear for natural language reference
- `first_frame_description`: A natural language description of where the object is located in the first frame of the Q_window (e.g., "the player in red jersey number 10 on the right side of the court"). We will use this to generate the initial bounding box.

The segment's starting frame in the original video is frame {num_first_frame}.
The segment has {total_frames} frames in total.

Please analyze the video content, identify the most prominent moving subject, and return your annotation in JSON format. Your annotation content should be in English. Make the query specific enough to uniquely identify the target, but general enough to be a natural referring expression.
