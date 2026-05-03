# Step 1 — Task Understanding

## Role
LLM call. Parses raw natural language into a structured JSON object.

## Why a separate step?
The LLM is given exactly one job: extract structure. If this was merged with
Step 3, the model would simultaneously try to understand the task AND reason
about model selection — producing worse output for both. Isolation forces
precision.

## System Prompt

```
You are an ML task analyst. Your ONLY job is to parse a user's 
natural language ML task description into a structured JSON object.

Extract exactly these fields:
- task_type: the ML task (e.g. image_classification, object_detection, 
  text_classification, regression, clustering, generation, segmentation, 
  named_entity_recognition, recommendation, anomaly_detection)
- data_type: the data modality (e.g. satellite_imagery, medical_images, 
  tabular, text, audio, time_series, video, point_cloud)
- scale: estimated dataset/deployment scale — "small" (<10k samples), 
  "medium" (10k–1M), or "large" (>1M)
- constraints: list of any mentioned constraints (e.g. limited compute, 
  real-time inference, edge deployment, interpretability required, 
  no labelled data). Empty list if none mentioned.
- domain: the application domain (e.g. remote_sensing, healthcare, 
  finance, ecommerce, nlp_research, robotics, general)

RULES:
1. Respond ONLY with valid JSON — no explanation, no markdown, no extra text.
2. If a field is not mentioned, make a reasonable inference from context.
3. Use lowercase_snake_case for all string values.
4. constraints must always be a list, even if empty.
```

## User Prompt Template

```
Parse this ML task description:

"{raw_input}"

Return only the JSON object.
```

## Key design decisions

- `temperature=0.1` — low temperature forces deterministic, consistent JSON output.
  Higher temperature caused field name variations (e.g. "taskType" vs "task_type")
  that broke JSON parsing downstream.
- The RULES block was added after testing showed the model sometimes wrapping
  output in markdown fences (```json ... ```) — rule 1 reduced this, and the
  fence-stripping fallback in code handles remaining cases.
- `lowercase_snake_case` constraint was added after early runs returned
  "Image Classification" (title case with space) which caused mismatches in
  Step 2's query builder.

## What Step 2 depends on from this output
- `task_type` → used to build the Tavily search query
- `data_type` → used to build the Tavily search query  
- `domain`    → used to build the Tavily search query
- `constraints` → passed to Step 3 to constrain model selection
- `scale`     → passed to Steps 3, 4, 5