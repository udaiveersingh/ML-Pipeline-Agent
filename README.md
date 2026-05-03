# ML Pipeline Agent

**Assignment:** Build a Multi-Step LLM Agent  
**Author:** Udaiveer Singh 

A 5-step LLM agent that takes a natural language ML task description, retrieves real research via Tavily, and outputs a complete, structured model selection and training pipeline report. Each step feeds into the next — no step can be skipped without breaking the chain.

## What it does

1. **Step 1 (LLM)** — Parses your task description into structured fields (task type, data type, scale, constraints, domain)
2. **Step 2 (TOOL)** — Searches Tavily API for real SOTA models and benchmarks relevant to your task
3. **Step 3 (LLM)** — Reasons over retrieved results to select the best model architecture with explicit justification
4. **Step 4 (LLM)** — Designs a complete, concrete training pipeline tailored to the selected model
5. **Step 5 (LLM)** — Identifies specific risks in the pipeline and recommends prioritised next steps

**Output:** Markdown report + JSON state dump with all intermediate results

## Quick Start

```bash
# Clone and enter directory
git clone https://github.com/your-username/ml-pipeline-agent.git
cd ml-pipeline-agent

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
echo "GROQ_API_KEY=your_groq_key" > .env
echo "TAVILY_API_KEY=your_tavily_key" >> .env

# Run the agent
python main.py "I want to classify satellite images into land use categories"
```

## Getting API Keys

**Groq API (free):**
1. Go to https://console.groq.com
2. Sign up (Google login works)
3. Click "API Keys" → "Create API Key"

**Tavily API (free):**
1. Go to https://app.tavily.com
2. Sign up
3. Copy API key from dashboard (1000 searches/month free tier)

## Usage

```bash
# Interactive mode — prompts for input
python main.py

# Direct input — pass task as argument
python main.py "detect fraud in financial transactions with limited compute and real-time inference requirements"
```

## Output Files

Each run creates unique timestamped outputs:
```
output/
├── report_2026-05-02T23-17-04.md      # Structured markdown report
├── state_2026-05-02T23-17-04.json     # Full agent state (all step results)
├── report_2026-05-02T23-24-48.md      # Second run
├── state_2026-05-02T23-24-48.json
└── ...
```

**Report sections:**
- Task Understanding (structured fields extracted)
- Retrieved SOTA Approaches (papers, articles, benchmarks)
- Model Selection (ranked candidates + chosen model + rationale)
- Training Pipeline (architecture, preprocessing, training config, evaluation metrics)
- Risks & Tradeoffs (specific risks with severity + mitigations)
- Recommended Next Steps (prioritised action items)

## Chain Architecture

```
User Input (raw string)
    ↓
[Step 1 - LLM]  task_type, data_type, scale, constraints, domain
    ↓
[Step 2 - TOOL] 6 SOTA results with titles, summaries, URLs
    ↓
[Step 3 - LLM]  3 candidate models ranked + selected model + rationale
    ↓
[Step 4 - LLM]  full pipeline spec: preprocessing, architecture, training, eval
    ↓
[Step 5 - LLM]  risks with severity + compute estimate + next steps
    ↓
[Report Writer] markdown report + JSON state dump
```

**Why no step can be skipped:**
- Step 3 needs Step 2's retrieved models (without them, model selection hallucinates)
- Step 4 needs Step 3's selected model (pipeline design is specific to that architecture)
- Step 5 needs Step 4's pipeline (risk analysis critiques those specific choices)
- Step 2 needs Step 1's structured fields (raw user text produces poor search queries)

## Project Structure

```
ml-pipeline-agent/
├── main.py                      # Chain runner — orchestrates all steps
├── state.py                     # Shared state dict + helpers
├── steps/
│   ├── __init__.py
│   ├── step1_understand.py      # LLM: parse task → structured JSON
│   ├── step2_search.py          # TOOL: Tavily search → SOTA results
│   ├── step3_select.py          # LLM: model selection reasoning
│   ├── step4_pipeline.py        # LLM: full pipeline design
│   ├── step5_risks.py           # LLM: risk analysis + next steps
│   └── report_writer.py         # Write final markdown + JSON
├── prompts/
│   ├── step1.md                 # Step 1 prompt + design rationale
│   ├── step2.md                 # Step 2 prompt + design rationale
│   ├── step3.md                 # Step 3 prompt + iteration history
│   └── step4_and_step5.md       # Steps 4 & 5 prompts + rationale
├── output/                      # Generated reports (timestamped)
├── requirements.txt
├── .env                         # Your API keys (add to .gitignore)
├── .gitignore
└── README.md
```

## Design Decisions

**Why multi-step chaining?**
A single LLM prompt cannot reliably parse intent, retrieve current research, reason about model tradeoffs, and design a full pipeline simultaneously. Each sub-task requires a different cognitive mode. Separating them allows each step to be prompted precisely for its job, with concrete input from the previous step grounding the next.

**Why Tavily instead of pure LLM?**
The LLM cannot know which models were published this month, what benchmark scores they currently have, or whether a paper was retracted. Tavily retrieves real, verifiable, current information — preventing hallucinated paper citations and fabricated accuracy numbers.

**Why separate Step 3 (model selection) from Step 4 (pipeline design)?**
Early testing merged them into one LLM call. The output was consistently vague — the model would suggest a model and provide generic pipeline advice without reasoning about why specific choices (optimizer, LR schedule, augmentation) were appropriate for that architecture. Separating them forced explicit comparative reasoning in Step 3 before designing specifics in Step 4.

## Error Handling

If an API call fails:
- **Step 1 JSON parse fails** → Defaults to `task_type='unknown'` and continues (degraded but functional)
- **Step 2 search returns nothing** → Logs error and continues; Step 3 detects empty results and falls back to training knowledge
- **Any step LLM call fails** → Error logged to `state["errors"]`, chain continues with defaults

See `state.py` for `record_error()` and `mark_step_done()` helpers.

## Demo & Evaluation

For the live demo:
1. Run: `python main.py "I want to classify satellite images into land use categories"`
2. Pause after each step and narrate what was input and output (check `state_dump.json`)
3. Show a second run with a different task to prove the chain adapts
4. Show the `prompts/` folder and explain why each prompt is shaped that way
5. Demonstrate graceful degradation by running with a bad API key

## Dependencies

- **groq** — LLM API calls (Llama 3.3 70B)
- **tavily-python** — SOTA search (Step 2 tool)
- **python-dotenv** — Load API keys from `.env`

## Limitations

- **Vague inputs** produce degraded chains (e.g., "blah blah I want to do something with data")
- **Niche domains** with sparse online coverage may return irrelevant Tavily results
- **Compute estimates** are LLM guesses, not validated calculations
- **Semi-supervised/few-shot scenarios** not explicitly handled despite being common constraints
- **JSON parsing** relies on LLM following schema; fallbacks exist but degrade experience

See the written report (`assignment_report.docx`) for full limitations and reflection.

## Author

Udaiveer Singh 
Plaksha University  