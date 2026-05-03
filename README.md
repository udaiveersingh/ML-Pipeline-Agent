# ML Pipeline Agent

A multi-step LLM agent that takes a natural language ML task description and produces a complete, structured model selection and training pipeline report.

## What it does

1. **Step 1 (LLM)** — Parses your task description into structured fields (task type, data type, scale, constraints)
2. **Step 2 (TOOL)** — Searches Tavily for real SOTA models and benchmarks relevant to your task
3. **Step 3 (LLM)** — Reasons over retrieved results to select the best model architecture
4. **Step 4 (LLM)** — Designs a complete training pipeline tailored to the selected model
5. **Step 5 (LLM)** — Identifies specific risks in the pipeline and recommends mitigations

## Setup

```bash
pip install groq tavily-python python-dotenv
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

## Run

```bash
# Interactive mode
python main.py

# Or pass input directly
python main.py "I want to detect tumours in MRI scans with limited labelled data"
```

## Output

- `output/report.md` — Full structured report
- `output/state_dump.json` — Complete agent state (all step outputs)

## Chain structure

Each step is a separate function in `steps/`. The state dict is passed through every step and accumulates results. No step can be skipped without breaking the chain:
- Step 3 needs Step 2's retrieved models
- Step 4 needs Step 3's selected model  
- Step 5 needs Step 4's designed pipeline