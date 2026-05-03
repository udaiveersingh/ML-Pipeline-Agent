"""
step4_pipeline.py — Training Pipeline Design

INPUT  : state["task_type"], state["data_type"], state["scale"],
         state["selected_model"], state["selection_rationale"]
OUTPUT : state["pipeline"] (preprocessing, architecture, training, evaluation)

Why this step cannot run without Step 3?
  The pipeline is designed around the SPECIFIC selected model.
  A ResNet pipeline (augmentation, SGD, step LR) differs significantly
  from a ViT pipeline (patch embeddings, AdamW, warmup schedule).
  Generic pipeline advice is useless — this step must know the model first.
"""

import json
from groq import Groq
from state import mark_step_done, record_error

SYSTEM_PROMPT = """You are an ML pipeline engineer. Given a selected model and 
task specification, design a complete, practical training pipeline.

Return a JSON object with exactly these keys:
- preprocessing: list of steps (each a string describing one operation)
- architecture: object with:
    - base_model: string
    - modifications: list of strings (e.g. "replace final FC layer with 10-class head")
    - input_size: string (e.g. "224x224x3")
    - pretrained: boolean
- training: object with:
    - optimizer: string
    - learning_rate: string (with schedule, e.g. "1e-4 with cosine decay")
    - batch_size: string (e.g. "32 (adjust for GPU memory)")
    - epochs: string (e.g. "50 with early stopping, patience=10")
    - loss_function: string
    - augmentation: list of strings
    - regularization: list of strings
- evaluation: list of metrics (each a string describing metric + threshold)

RULES:
1. Respond ONLY with valid JSON.
2. Be specific and practical — mention actual values, not vague guidance.
3. Tailor ALL choices to the selected model and constraints.
4. For pretrained=true, recommend fine-tuning strategy.
5. evaluation metrics must be appropriate for the task type.
"""


def step4_pipeline(state: dict, api_key: str) -> dict:
    """
    Design the full training pipeline for the selected model.
    Writes: state["pipeline"]
    """
    client = Groq(api_key=api_key)

    constraints_str = ", ".join(state["constraints"]) if state["constraints"] else "none"

    user_prompt = f"""Design a pipeline for:
- Task         : {state['task_type']}
- Data type    : {state['data_type']}
- Scale        : {state['scale']} dataset
- Selected model: {state['selected_model']}
- Why selected : {state['selection_rationale']}
- Constraints  : {constraints_str}

Return the complete pipeline JSON object."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        state["pipeline"] = {
            "preprocessing": parsed.get("preprocessing", []),
            "architecture":  parsed.get("architecture", {}),
            "training":      parsed.get("training", {}),
            "evaluation":    parsed.get("evaluation", []),
        }

        mark_step_done(state, "step4_pipeline")

    except json.JSONDecodeError as e:
        record_error(state, "step4_pipeline", f"JSON parse failed: {e}")

    except Exception as e:
        record_error(state, "step4_pipeline", str(e))

    return state