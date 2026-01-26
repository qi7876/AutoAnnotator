You are an expert video data annotator.

I will provide you with a single frame from a sports video (10 FPS). You need to annotate a task called "Objects Spatial Relationships". Your task is to create a question about the spatial relationship between two or more objects in the frame.

Please annotate according to the following JSON template:

```json
{{
    "annotation_id": "1",
    "task_L1": "Understanding",
    "task_L2": "Objects_Spatial_Relationships",
    "timestamp_frame": 100,
    "question": "In which direction is the player in the red jersey (Player A) relative to the player in the blue jersey (Player B)?",
    "answer": "The player in the red jersey (Player A) is to the left of the player in the blue jersey (Player B).",
    "bounding_box": [
        {{
            "label": "Player A",
            "description": "the player wearing red jersey number 10"
        }},
        {{
            "label": "Player B",
            "description": "the player wearing blue jersey number 5"
        }}
    ]
}}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will maintain these fields, you don't need to annotate them
- `timestamp_frame`: The frame number in the clip where the spatial relationship is observed. Use 0-indexed clip frame numbers. Valid range: 0..{max_frame} (inclusive).
- `question`: Create a question about the spatial relationship between two objects (e.g., left/right, front/back, above/below)
- `answer`: Provide a clear answer describing the spatial relationship
- `bounding_box`: For each object, provide:
  - `label`: A short label for the object (e.g., "Player A", "Ball", "Goal")
  - `description`: A natural language description of the object that will be used for grounding (e.g., "the player wearing red jersey number 10")

The segment has {total_frames} frames in total.
Valid clip frame indices: 0..{max_frame} (inclusive).

Please analyze the frame content and return your annotation in JSON format. Choose 2 prominent objects in the frame for spatial relationship annotation.
