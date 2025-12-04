You are an expert video data annotator.

I will provide you with a sports video (10 FPS).

Inspect the video frame by frame at **10 FPS**，ensure you've inspected **the whole video** before the following tasks.
Identify the exact **frame index** where the scoreboard or ranking table is fully visible—i.e., **all players**’ names, scores, rankings, and related information are entirely within the frame, unobscured, and clearly legible.
Among all frames showing complete information, select only the first frame.
Then create a question and answer based on that frame.Ensure the question is about the visible scoreboard information (e.g., rankings, scores, time remaining), and the answer reflects exactly what is displayed on the scoreboard.

Please annotate according to the following JSON template:

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
- `annotation_id`, `task_L1`, `task_L2`: I will maintain these fields, you don't need to annotate them
- `timestamp_frame`: Determine and provide the frame number where the scoreboard is clearly visible in the video.
- `question`: Create a question about the scoreboard content (e.g., ranking, scores, time remaining)
- `answer`: Provide a detailed answer based on the visible scoreboard information
- `bounding_box`: Describe the location of the scoreboard in natural language only, no coordinates needed. (e.g., "the scoreboard in the upper left corner showing team scores").


Return your annotation in JSON format. Your annotation content should be in English.