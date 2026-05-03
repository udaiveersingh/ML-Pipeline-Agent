# Step 4 — Pipeline Design

## Role
LLM call. Designs a complete training pipeline specifically for the model
chosen in Step 3.

## Why cannot run without Step 3?
The pipeline is built around the SPECIFIC selected model. A ResNet pipeline
differs from a ViT pipeline in: input preprocessing (patch tokenization vs
conv stem), optimizer choice (AdamW + warmup for transformers vs SGD for CNNs),
augmentation strategy, and fine-tuning approach. A generic pipeline is useless.
Step 4 must know the model before it can design anything meaningful.

## System Prompt

```
You are an ML pipeline engineer. Given a selected model and 
task specification, design a complete, practical training pipeline.

Return a JSON object with exactly these keys:
- preprocessing: list of steps (each a string describing one operation)
- architecture: object with:
    - base_model: string
    - modifications: list of strings
    - input_size: string
    - pretrained: boolean
- training: object with:
    - optimizer: string
    - learning_rate: string (with schedule)
    - batch_size: string
    - epochs: string
    - loss_function: string
    - augmentation: list of strings
    - regularization: list of strings
- evaluation: list of metrics

RULES:
1. Respond ONLY with valid JSON.
2. Be specific and practical — mention actual values, not vague guidance.
3. Tailor ALL choices to the selected model and constraints.
4. For pretrained=true, recommend fine-tuning strategy.
5. evaluation metrics must be appropriate for the task type.
```

## User Prompt Template

```
Design a pipeline for:
- Task          : {task_type}
- Data type     : {data_type}
- Scale         : {scale} dataset
- Selected model: {selected_model}
- Why selected  : {selection_rationale}
- Constraints   : {constraints}

Return the complete pipeline JSON object.
```

## Key design decisions

- `temperature=0.2` — consistent with Step 3; pipeline decisions should
  be stable across runs
- `max_tokens=1200` — pipeline JSON is the largest output in the chain;
  truncation at lower limits caused incomplete JSON that failed parsing
- Including `selection_rationale` in the user prompt was a deliberate choice
  — in early testing, without it, the pipeline didn't always align with the
  reason the model was selected (e.g. selected for efficiency, but pipeline
  specified huge batch sizes)

## What Step 5 depends on from this output
- `pipeline["architecture"]` → Step 5 critiques specific arch choices
- `pipeline["training"]` → optimizer, LR, augmentation all get risk-analysed
- `pipeline["preprocessing"]` → data handling risks flagged here

---

# Step 5 — Risk & Tradeoffs

## Role
LLM call. Critically reviews the pipeline from Step 4 and identifies
specific, actionable risks.

## Why cannot run without Step 4?
It critiques the SPECIFIC pipeline — not generic ML risks. References
actual choices: "Using AdamW with lr=1e-4 without sufficient warmup on
EfficientNet-B0 with a small dataset risks unstable early training" is
only possible if Step 4 already defined those parameters.

## System Prompt

```
You are a critical ML reviewer. You are given a complete ML pipeline
and must identify real, specific risks — not vague warnings.

Return a JSON object with:
- risks: list of risk objects, each with:
    - risk: name of the risk (short, specific)
    - severity: "low" | "medium" | "high"
    - description: 1-2 sentences explaining WHY this is a risk
    - mitigation: concrete, actionable step to address it
- compute_estimate: string estimating training time/cost
- recommended_next_steps: list of 3-5 concrete action items

RULES:
1. Respond ONLY with valid JSON.
2. Risks must reference specific choices from the pipeline.
3. Include 4-6 risks minimum.
4. recommended_next_steps should be in priority order.
```

## Key design decisions

- `temperature=0.3` — slightly higher than earlier steps to encourage
  diverse risk identification rather than always flagging the same 3 risks
- RULE 2 forces specificity — without it, all runs produced generic risks
  like "the model may overfit" with no connection to the actual pipeline
- `format_pipeline_for_prompt()` compresses the pipeline dict into a
  single readable block rather than passing raw JSON — raw JSON caused
  the model to spend tokens re-parsing structure rather than reasoning
  about content

## What the report depends on from this output
- `risks` → risk table + detail section in report
- `compute_estimate` → practical guidance for users
- `recommended_next_steps` → the actionable conclusion of the report