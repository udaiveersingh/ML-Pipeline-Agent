"""
step5_risks.py — Risk Analysis & Tradeoffs

INPUT  : state["pipeline"], state["selected_model"], state["constraints"],
         state["task_type"], state["scale"]
OUTPUT : state["risks"], state["compute_estimate"],
         state["recommended_next_steps"]

Why this step cannot run without Step 4?
  It critiques the SPECIFIC pipeline designed in Step 4 — not generic ML risks.
  It references actual choices (optimizer, augmentation, architecture) 
  and explains why THOSE choices might fail with THOSE constraints.
"""

import json
from groq import Groq
from state import mark_step_done, record_error

SYSTEM_PROMPT = """You are a critical ML reviewer. You are given a complete ML pipeline
and must identify real, specific risks — not vague warnings.

Return a JSON object with:
- risks: list of risk objects, each with:
    - risk: name of the risk (short, specific)
    - severity: "low" | "medium" | "high"
    - description: 1-2 sentences explaining WHY this is a risk for this pipeline
    - mitigation: concrete, actionable step to address it
- compute_estimate: string estimating training time/cost 
  (e.g. "~4 hours on a single A100 for 50 epochs at batch_size=32")
- recommended_next_steps: list of 3-5 concrete action items for the user
  (e.g. "Run a baseline with frozen backbone for 5 epochs before full fine-tuning")

RULES:
1. Respond ONLY with valid JSON.
2. Risks must reference specific choices from the pipeline — e.g. 
   "Using AdamW with lr=1e-4 without warmup on a ViT can destabilise 
   early training" not "overfitting may occur."
3. Include 4-6 risks minimum.
4. recommended_next_steps should be in priority order.
"""


def format_pipeline_for_prompt(pipeline: dict) -> str:
    """Summarise the pipeline into a compact string for the prompt."""
    arch     = pipeline.get("architecture", {})
    training = pipeline.get("training", {})
    prep     = pipeline.get("preprocessing", [])
    evals    = pipeline.get("evaluation", [])

    lines = [
        f"Architecture  : {arch.get('base_model')} | pretrained={arch.get('pretrained')}",
        f"Modifications : {'; '.join(arch.get('modifications', []))}",
        f"Input size    : {arch.get('input_size')}",
        f"Optimizer     : {training.get('optimizer')}",
        f"Learning rate : {training.get('learning_rate')}",
        f"Batch size    : {training.get('batch_size')}",
        f"Epochs        : {training.get('epochs')}",
        f"Loss          : {training.get('loss_function')}",
        f"Augmentation  : {'; '.join(training.get('augmentation', []))}",
        f"Regularization: {'; '.join(training.get('regularization', []))}",
        f"Preprocessing : {'; '.join(prep[:4])}",
        f"Evaluation    : {'; '.join(evals[:4])}",
    ]
    return "\n".join(lines)


def step5_risks(state: dict, api_key: str) -> dict:
    """
    Identify risks and next steps for the designed pipeline.
    Writes: risks, compute_estimate, recommended_next_steps
    """
    client = Groq(api_key=api_key)

    pipeline_summary = format_pipeline_for_prompt(state["pipeline"])
    constraints_str  = ", ".join(state["constraints"]) if state["constraints"] else "none"

    user_prompt = f"""Review this ML pipeline for risks:

Task    : {state['task_type']} on {state['data_type']}
Model   : {state['selected_model']}
Scale   : {state['scale']}
Constraints: {constraints_str}

Pipeline summary:
{pipeline_summary}

Return the risk analysis JSON."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        state["risks"]                  = parsed.get("risks", [])
        state["compute_estimate"]       = parsed.get("compute_estimate")
        state["recommended_next_steps"] = parsed.get("recommended_next_steps", [])

        mark_step_done(state, "step5_risks")

    except json.JSONDecodeError as e:
        record_error(state, "step5_risks", f"JSON parse failed: {e}")

    except Exception as e:
        record_error(state, "step5_risks", str(e))

    return state