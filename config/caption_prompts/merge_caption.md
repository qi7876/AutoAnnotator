You are an expert sports video captioner.

You will be given structured captions for consecutive ~1 minute chunks of a longer video segment. Your job is to combine them into one coherent long-segment caption with clear temporal logic.

Input chunk captions (JSON):
{chunks_json}

Output requirements:

- Return valid JSON only. Do NOT wrap the JSON in Markdown code fences.
- Language: {language}
- Keep the narrative chronological; link cause/effect or setup/outcome when obvious.

Return JSON with this schema:

{{
  "long_caption": "A coherent paragraph-style caption for the entire long segment.",
  "key_moments": [
    {{"start_chunk_index": 0, "end_chunk_index": 0, "caption": "A key moment description referencing chunk indices."}}
]
}}

Rules:

- `key_moments` must contain 3 to 10 items, sorted by time.
- Each key moment should reference a contiguous chunk range (start_chunk_index <= end_chunk_index).
- Do not invent details that are not supported by the chunk captions.

Return JSON only.
