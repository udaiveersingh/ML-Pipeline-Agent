"""
state.py — Shared state object for the ML Pipeline Agent.

Every step in the chain reads from and writes to this dictionary.
At the end of the chain it contains everything the agent has learned.
"""

import json
from datetime import datetime


def create_state(user_input: str) -> dict:
    """
    Initialise a fresh state dict for a new agent run.
    Only the raw user input is set at start — everything else
    gets filled in as the chain progresses.
    """
    return {
        # ── Input ────────────────────────────────────────────────
        "raw_input": user_input,
        "timestamp": datetime.now().isoformat(),

        # ── Step 1 output (Task Understanding) ───────────────────
        "task_type": None,        # e.g. "image_classification"
        "data_type": None,        # e.g. "satellite_imagery"
        "scale": None,            # e.g. "small" / "medium" / "large"
        "constraints": [],        # e.g. ["limited compute", "real-time inference"]
        "domain": None,           # e.g. "remote sensing", "medical imaging"

        # ── Step 2 output (SOTA Search — tool call) ───────────────
        "search_query": None,     # the query we sent to Tavily
        "sota_results": [],       # list of {title, summary, url}

        # ── Step 3 output (Model Selection) ──────────────────────
        "candidate_models": [],   # shortlisted models with pros/cons
        "selected_model": None,   # final pick
        "selection_rationale": None,

        # ── Step 4 output (Pipeline Design) ──────────────────────
        "pipeline": {
            "preprocessing": [],
            "architecture": {},
            "training": {},
            "evaluation": [],
        },

        # ── Step 5 output (Risk & Tradeoffs) ─────────────────────
        "risks": [],              # list of {risk, severity, mitigation}
        "compute_estimate": None,
        "recommended_next_steps": [],

        # ── Meta ──────────────────────────────────────────────────
        "steps_completed": [],    # track which steps ran successfully
        "errors": [],             # any step errors go here
    }


def mark_step_done(state: dict, step_name: str) -> None:
    """Record that a step completed successfully."""
    state["steps_completed"].append(step_name)
    print(f"  ✓ {step_name} complete")


def record_error(state: dict, step_name: str, error: str) -> None:
    """Log an error without crashing the chain."""
    state["errors"].append({"step": step_name, "error": error})
    print(f"  ✗ {step_name} failed: {error}")


def print_state(state: dict) -> None:
    """Pretty-print the current state — useful during demo."""
    print("\n" + "=" * 60)
    print("  CURRENT AGENT STATE")
    print("=" * 60)
    print(json.dumps(state, indent=2, default=str))
    print("=" * 60 + "\n")


def save_state(state: dict, path: str = "output/state_dump.json") -> None:
    """Save full state to JSON — useful for debugging and the report."""
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"  State saved to {path}")