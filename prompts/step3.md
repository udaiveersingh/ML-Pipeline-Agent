# Step 3 — Model Selection Reasoning

## Role
LLM call. Reasons over Step 1's task spec + Step 2's retrieved results
to select one model and justify the decision.

## Why cannot run without Step 2?
The system prompt injects actual retrieved paper titles and summaries.
The LLM is comparing real candidates found in the literature — not
generating generic suggestions from training data. If Step 2 returns
nothing, the model explicitly notes the fallback. The comparison is
grounded, not hallucinated.

## System Prompt

```
You are a senior ML engineer advising on model selection.
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
```

## User Prompt Template

```
Task details:
- Task type    : {task_type}
- Data type    : {data_type}
- Scale        : {scale}
- Constraints  : {constraints}
- Domain       : {domain}

Retrieved SOTA results:
{formatted_sota_block}

Select the best model architecture and return the JSON object.
```

## Key design decisions

- `temperature=0.2` — slightly higher than Step 1 to allow reasoning
  variation, but still structured enough for reliable JSON
- RULE 3 (respect constraints) was added after a test run where the model
  recommended ViT-Large for a "limited_compute" task — the constraint was
  being ignored when the retrieved papers showed transformers performing well
- `benchmark_ref` field forces the model to cite its source for each
  candidate, making the reasoning auditable
- Only top 5 of 6 results are injected (`.[:5]`) — the 6th result in
  testing was consistently low-relevance and adding it caused the model
  to include irrelevant candidates

## Prompt iteration example

**Version 1 (didn't work):**
```
Given this task and these papers, pick the best model.
```
Problem: returned a paragraph of text with the model name buried inside.
Could not parse reliably.

**Version 2 (still broken):**
```
Return JSON with selected_model and rationale.
```
Problem: sometimes returned {"selected_model": "EfficientNet", "rationale": "..."}
and sometimes {"model": "EfficientNet", "reason": "..."} — inconsistent keys
broke downstream state writes.

**Version 3 (current — works):**
Full schema defined in system prompt with exact field names + RULES block
enforcing JSON-only output. Adding `candidate_models` array forced the model
to do explicit comparison before selecting, improving selection quality.

## What Step 4 depends on from this output
- `selected_model` → the model Step 4 designs the pipeline around
- `selection_rationale` → injected into Step 4's prompt so the pipeline
  design is aware of WHY this model was chosen
- `candidate_models` → shown in the final report for transparency