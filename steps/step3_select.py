"""
step3_select.py — Model Selection Reasoning

INPUT  : state["task_type"], state["data_type"], state["scale"],
         state["constraints"], state["sota_results"]
OUTPUT : state["candidate_models"], state["selected_model"],
         state["selection_rationale"]

Why a separate step from Step 4?
  Selection reasoning (WHICH model) is a different cognitive task from
  pipeline design (HOW to train it). Mixing them produces vague output
  for both. Step 3 focuses purely on the decision and its justification.

Why this step cannot run without Step 2?
  The system prompt injects actual retrieved papers/models. The LLM is
  comparing real candidates, not generating generic suggestions.
"""

import json
from groq import Groq
from state import mark_step_done, record_error

SYSTEM_PROMPT = """You are a senior ML engineer advising on model selection.
You will be given a task description and a list of SOTA approaches retrieved 
from recent papers and benchmarks.

Your job is to reason carefully and return a JSON object with:
- candidate_models: list of 3 models worth considering, each with:
    - name: model name
    - type: architecture type (CNN, Transformer, RNN, hybrid, etc.)
    - pros: list of 2-3 advantages for this task
    - cons: list of 1-2 disadvantages
    - benchmark_ref: title of the source result (or "general knowledge")
- selected_model: the single best choice (name string)
- selection_rationale: 2-3 sentences explaining WHY this model wins 
  given the specific constraints and data type

RULES:
1. Respond ONLY with valid JSON — no markdown, no explanation outside JSON.
2. Base your reasoning on the retrieved results where possible.
3. Always respect constraints — if limited_compute is a constraint,
   do NOT recommend a ViT-Large or GPT-scale model.
4. If search results are empty, use your training knowledge but note it.
"""


def format_sota_for_prompt(sota_results: list) -> str:
    """Format retrieved results into a readable block for the LLM."""
    if not sota_results:
        return "No search results available. Use general ML knowledge."

    lines = []
    for i, r in enumerate(sota_results[:5], 1):   # cap at 5
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['summary'][:300]}")
        lines.append(f"   Source: {r['url']}\n")
    return "\n".join(lines)


def step3_select(state: dict, api_key: str) -> dict:
    """
    Select the best model by reasoning over task requirements + SOTA results.
    Writes: candidate_models, selected_model, selection_rationale
    """
    client = Groq(api_key=api_key)

    sota_block = format_sota_for_prompt(state["sota_results"])
    constraints_str = ", ".join(state["constraints"]) if state["constraints"] else "none specified"

    user_prompt = f"""Task details:
- Task type    : {state['task_type']}
- Data type    : {state['data_type']}
- Scale        : {state['scale']}
- Constraints  : {constraints_str}
- Domain       : {state['domain']}

Retrieved SOTA results:
{sota_block}

Select the best model architecture and return the JSON object."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=900,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        state["candidate_models"]     = parsed.get("candidate_models", [])
        state["selected_model"]       = parsed.get("selected_model")
        state["selection_rationale"]  = parsed.get("selection_rationale")

        mark_step_done(state, "step3_select")

    except json.JSONDecodeError as e:
        record_error(state, "step3_select", f"JSON parse failed: {e}")
        state["selected_model"]      = "Unable to determine"
        state["selection_rationale"] = "Step failed — see errors."

    except Exception as e:
        record_error(state, "step3_select", str(e))

    return state