"""
step1_understand.py — Task Understanding

INPUT  : state["raw_input"]  (raw user string)
OUTPUT : state fields: task_type, data_type, scale, constraints, domain

Why a separate step?
  The LLM is given ONE job here: extract structure from messy natural language.
  Keeping this isolated means Step 3 always receives clean, consistent fields
  regardless of how the user phrased their input.
"""

import json
from groq import Groq
from state import mark_step_done, record_error

SYSTEM_PROMPT = """You are an ML task analyst. Your ONLY job is to parse a user's 
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

Example output:
{
  "task_type": "image_classification",
  "data_type": "satellite_imagery",
  "scale": "medium",
  "constraints": ["limited_compute", "high_accuracy_required"],
  "domain": "remote_sensing"
}"""


def step1_understand(state: dict, api_key: str) -> dict:
    """
    Parse the user's raw input into structured task fields.
    Writes: task_type, data_type, scale, constraints, domain
    """
    client = Groq(api_key=api_key)

    user_prompt = f"""Parse this ML task description:

\"{state['raw_input']}\"

Return only the JSON object."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,   # low temp = consistent structured output
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        # Write parsed fields into state
        state["task_type"]   = parsed.get("task_type")
        state["data_type"]   = parsed.get("data_type")
        state["scale"]       = parsed.get("scale", "medium")
        state["constraints"] = parsed.get("constraints", [])
        state["domain"]      = parsed.get("domain")

        mark_step_done(state, "step1_understand")

    except json.JSONDecodeError as e:
        record_error(state, "step1_understand", f"JSON parse failed: {e} | raw: {raw}")
        # Fallback defaults so chain can continue
        state["task_type"]   = "unknown"
        state["data_type"]   = "unknown"
        state["scale"]       = "medium"
        state["constraints"] = []
        state["domain"]      = "general"

    except Exception as e:
        record_error(state, "step1_understand", str(e))

    return state