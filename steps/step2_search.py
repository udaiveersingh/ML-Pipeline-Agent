"""
step2_search.py — SOTA Retrieval (TOOL CALL)

INPUT  : state["task_type"], state["data_type"], state["domain"]
OUTPUT : state["search_query"], state["sota_results"]

Why a tool call here instead of LLM?
  The LLM cannot reliably know which models were published last month,
  what benchmark scores are current, or which papers exist.
  Tavily retrieves real, up-to-date information — grounding Step 3's
  reasoning in actual evidence rather than hallucination.
"""

from tavily import TavilyClient
from state import mark_step_done, record_error


def build_search_query(state: dict) -> str:
    """
    Build a targeted search query from Step 1's structured output.
    This is more precise than a raw user string.
    """
    task   = state.get("task_type", "machine learning")
    data   = state.get("data_type", "")
    domain = state.get("domain", "")

    # Build a query that targets benchmarks and SOTA papers
    parts = [f"best deep learning models for {task}"]
    if data and data != "unknown":
        parts.append(f"on {data.replace('_', ' ')}")
    if domain and domain not in ("general", "unknown"):
        parts.append(f"in {domain.replace('_', ' ')}")
    parts.append("SOTA benchmark 2024")

    return " ".join(parts)


def parse_results(raw_results: list) -> list:
    """
    Extract clean, structured records from Tavily's response.
    Only keep fields that are useful for Step 3.
    """
    parsed = []
    for r in raw_results:
        parsed.append({
            "title":   r.get("title", "Untitled"),
            "summary": r.get("content", "")[:500],  # cap at 500 chars
            "url":     r.get("url", ""),
            "score":   round(r.get("score", 0.0), 3),  # Tavily relevance score
        })
    # Sort by relevance score descending
    parsed.sort(key=lambda x: x["score"], reverse=True)
    return parsed


def step2_search(state: dict, api_key: str) -> dict:
    """
    Search for SOTA models and approaches using Tavily.
    Writes: search_query, sota_results
    """
    client = TavilyClient(api_key=api_key)

    query = build_search_query(state)
    state["search_query"] = query

    try:
        response = client.search(
            query=query,
            max_results=6,
            search_depth="advanced",   # deeper = better results
            include_answer=True,       # Tavily's AI summary of results
        )

        raw_results = response.get("results", [])

        if not raw_results:
            record_error(state, "step2_search", "Tavily returned no results")
            state["sota_results"] = []
            return state

        state["sota_results"] = parse_results(raw_results)

        # Bonus: store Tavily's own AI-generated answer if available
        if response.get("answer"):
            state["tavily_summary"] = response["answer"]

        mark_step_done(state, "step2_search")

    except Exception as e:
        record_error(state, "step2_search", str(e))
        # Graceful degradation — Step 3 will note that search failed
        state["sota_results"] = []

    return state