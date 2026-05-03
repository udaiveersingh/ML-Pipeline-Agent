"""
main.py — Entry point for the ML Pipeline Agent.

Runs the full 5-step chain:
  Step 1 (LLM)  → Task understanding
  Step 2 (TOOL) → SOTA search via Tavily
  Step 3 (LLM)  → Model selection reasoning
  Step 4 (LLM)  → Pipeline design
  Step 5 (LLM)  → Risk & tradeoffs
  → Final report written to output/report.md
"""

import os
import sys
from dotenv import load_dotenv

from state import create_state, print_state, save_state
from steps.step1_understand import step1_understand
from steps.step2_search import step2_search
from steps.step3_select import step3_select
from steps.step4_pipeline import step4_pipeline
from steps.step5_risks import step5_risks
from steps.report_writer import write_report

# ── Load environment variables from .env ─────────────────────────
load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found. Check your .env file.")
    sys.exit(1)
if not TAVILY_API_KEY:
    print("ERROR: TAVILY_API_KEY not found. Check your .env file.")
    sys.exit(1)


def run_agent(user_input: str, verbose: bool = False) -> dict:
    """
    Run the full agent chain for a given user input.
    Returns the final state dict.
    """
    print("\n" + "=" * 60)
    print("  ML PIPELINE AGENT")
    print("=" * 60)
    print(f"  Input: {user_input}\n")

    # ── Initialise shared state ───────────────────────────────────
    state = create_state(user_input)

    # ── Step 1: Task Understanding ────────────────────────────────
    print("[ Step 1 ] Task Understanding...")
    state = step1_understand(state, GROQ_API_KEY)
    if verbose:
        print(f"    task_type  : {state['task_type']}")
        print(f"    data_type  : {state['data_type']}")
        print(f"    scale      : {state['scale']}")
        print(f"    constraints: {state['constraints']}")

    # ── Step 2: SOTA Search (tool call) ──────────────────────────
    print("\n[ Step 2 ] Searching for SOTA approaches (tool call)...")
    state = step2_search(state, TAVILY_API_KEY)
    if verbose:
        print(f"    query      : {state['search_query']}")
        print(f"    results    : {len(state['sota_results'])} papers/articles found")

    # ── Step 3: Model Selection ───────────────────────────────────
    print("\n[ Step 3 ] Model Selection Reasoning...")
    state = step3_select(state, GROQ_API_KEY)
    if verbose:
        print(f"    selected   : {state['selected_model']}")

    # ── Step 4: Pipeline Design ───────────────────────────────────
    print("\n[ Step 4 ] Designing the Training Pipeline...")
    state = step4_pipeline(state, GROQ_API_KEY)
    if verbose:
        steps = state["pipeline"].get("preprocessing", [])
        print(f"    preprocessing steps: {len(steps)}")

    # ── Step 5: Risk Analysis ─────────────────────────────────────
    print("\n[ Step 5 ] Analysing Risks & Tradeoffs...")
    state = step5_risks(state, GROQ_API_KEY)
    if verbose:
        print(f"    risks identified: {len(state['risks'])}")

    # ── Write final report ────────────────────────────────────────
    print("\n[ Output ] Writing structured report...")
    os.makedirs("output", exist_ok=True)
    # Unique filename per run so previous outputs are never overwritten
    ts = state["timestamp"].replace(":", "-").replace(".", "-")[:19]
    report_path = f"output/report_{ts}.md"
    state_path  = f"output/state_{ts}.json"
    write_report(state, report_path)
    save_state(state, state_path)

    print("\n" + "=" * 60)
    print(f"  DONE — check {report_path}")

    return state


if __name__ == "__main__":
    # Accept input from command line or prompt interactively
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("Describe your ML task (e.g. 'I want to classify satellite images'):")
        user_input = input("> ").strip()

    if not user_input:
        print("No input provided. Exiting.")
        sys.exit(1)

    final_state = run_agent(user_input, verbose=True)