You are an expert video data annotator.

I will provide you with a video segment (10 FPS, you need to convert seconds to frame numbers). You need to annotate a task called "Continuous Actions Caption". Your task is to extract a series of actions performed by an athlete and annotate them according to the JSON template below.

Please annotate according to the following JSON template:

```json
{{
    "annotation_id": "1",
    "task_L1": "Understanding",
    "task_L2": "Continuous_Actions_Caption",
    "Q_window_frame": [10, 20],
    "question": "Please continuously describe the actions of the athlete in the video, and output the corresponding time intervals.",
    "A_window_frame": ["10-13", "14-17", "18-20"],
    "first_frame_description": "the player wearing red jersey number 10 on the right side of the court",
    "answer": [
        "Dribbles the ball to the left side for a breakthrough",
        "Performs a sudden stop and change of direction, shaking off the defender",
        "Jumps and takes a shot"
    ]
}}
```

Instructions:
- `annotation_id`, `task_L1`, `task_L2`: I will maintain these fields, you don't need to annotate them
- `Q_window_frame`: The question frame window, in original video frame numbers. Modify according to video content.
- `question`: Modify according to video content to specify which athlete's actions to describe
- `A_window_frame`: Answer frame windows, in original video frame numbers. The time segments should be continuous and non-overlapping.
- `first_frame_description`: A concise description of the target athlete in the first frame of Q_window_frame. Include appearance and location so the target can be grounded.
- `answer`: The answers should align with the number of answer frame windows. Each answer describes a distinct action phase.

The segment's starting frame in the original video is frame {num_first_frame}.
The segment has {total_frames} frames in total.

Please note the response format. The time periods in A_window_frame are segmented, and the number of segments should align with the number of answers in the answer array.

Please analyze the video content, combine it with the JSON template format, and return your annotation results.
