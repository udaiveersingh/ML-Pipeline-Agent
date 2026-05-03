# Step 2 — SOTA Retrieval (Tool Call)

## Role
External tool call — NOT an LLM call. Uses Tavily search API.

## Why a tool, not an LLM?
The LLM cannot reliably know:
- Which models were published in the last 6 months
- Current benchmark scores on specific datasets
- Whether a paper has been retracted or superseded

Asking the LLM to produce this information results in hallucinated paper titles
and fabricated accuracy numbers. Tavily retrieves real, verifiable, current
information — grounding Step 3's reasoning in actual evidence.

## Query Construction

The query is NOT the user's raw input. It is built programmatically from
Step 1's structured output:

```python
query = f"best deep learning models for {task_type} on {data_type} in {domain} SOTA benchmark 2024"
```

Example:
- Input fields: task_type="image_classification", data_type="satellite_imagery", domain="remote_sensing"
- Generated query: "best deep learning models for image_classification on satellite imagery in remote sensing SOTA benchmark 2024"

## Why this query shape?
- Including "SOTA benchmark 2024" biases results toward recent, comparative papers
  rather than tutorials or blog posts
- Using structured fields (not raw user text) avoids noisy queries like
  "best model for classifying my satellite pics" which returns worse results
- "deep learning" scopes out classical ML results that the agent can't pipeline

## Tavily API parameters

```python
client.search(
    query=query,
    max_results=6,
    search_depth="advanced",  # deeper crawl, better quality
    include_answer=True,      # returns Tavily's own AI summary
)
```

- `search_depth="advanced"` was chosen after testing showed "basic" depth
  returning low-quality blog content instead of papers
- `max_results=6` balances context window size vs coverage — more than 6
  causes Step 3's prompt to exceed useful context

## What Step 3 depends on from this output
- `sota_results` → list of {title, summary, url, score} — injected directly
  into Step 3's prompt for model comparison
- `search_query` → logged in state for transparency and the final report
- `tavily_summary` → Tavily's AI-generated synthesis, shown in the report

## Failure handling
If Tavily returns no results (network error, rate limit, bad query), the step
logs an error and sets `sota_results = []`. Step 3 detects the empty list
and falls back to training knowledge, explicitly noting the fallback in output.