"""
report_writer.py — Final Structured Report Generator

INPUT  : complete state dict (all 5 steps done)
OUTPUT : output/report.md

This is what a real user would actually read and act on.
It synthesises every step's output into a single coherent document.
"""

from state import mark_step_done


def write_report(state: dict, path: str = "output/report.md") -> None:
    """Generate a structured Markdown report from the final agent state."""

    lines = []

    # ── Header ────────────────────────────────────────────────────
    lines += [
        "# ML Pipeline Design Report",
        "",
        f"**Task:** {state.get('raw_input', 'N/A')}",
        f"**Generated:** {state.get('timestamp', 'N/A')}",
        "",
        "---",
        "",
    ]

    # ── Section 1: Task Summary ───────────────────────────────────
    lines += [
        "## 1. Task Understanding",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Task type | `{state.get('task_type', 'N/A')}` |",
        f"| Data type | `{state.get('data_type', 'N/A')}` |",
        f"| Scale | `{state.get('scale', 'N/A')}` |",
        f"| Domain | `{state.get('domain', 'N/A')}` |",
        f"| Constraints | {', '.join(f'`{c}`' for c in state.get('constraints', [])) or 'None specified'} |",
        "",
    ]

    # ── Section 2: SOTA Results ───────────────────────────────────
    lines += [
        "## 2. Retrieved SOTA Approaches",
        "",
        f"*Search query: `{state.get('search_query', 'N/A')}`*",
        "",
    ]
    sota = state.get("sota_results", [])
    if sota:
        for i, r in enumerate(sota[:5], 1):
            lines += [
                f"**{i}. {r['title']}**",
                f"> {r['summary'][:300]}",
                f"[Source]({r['url']})" if r.get('url') else "",
                "",
            ]
    else:
        lines += ["*No results retrieved — model selection based on training knowledge.*", ""]

    if state.get("tavily_summary"):
        lines += [
            "**AI Summary of retrieved results:**",
            f"> {state['tavily_summary']}",
            "",
        ]

    # ── Section 3: Model Selection ────────────────────────────────
    lines += [
        "## 3. Model Selection",
        "",
        f"**Selected model: `{state.get('selected_model', 'N/A')}`**",
        "",
        f"{state.get('selection_rationale', 'N/A')}",
        "",
        "### Candidates Considered",
        "",
    ]

    for m in state.get("candidate_models", []):
        lines += [
            f"#### {m.get('name')} ({m.get('type')})",
            f"- **Pros:** {', '.join(m.get('pros', []))}",
            f"- **Cons:** {', '.join(m.get('cons', []))}",
            f"- **Reference:** {m.get('benchmark_ref', 'general knowledge')}",
            "",
        ]

    # ── Section 4: Pipeline Design ────────────────────────────────
    pipeline = state.get("pipeline", {})
    arch     = pipeline.get("architecture", {})
    training = pipeline.get("training", {})

    lines += [
        "## 4. Training Pipeline",
        "",
        "### Architecture",
        f"- **Base model:** {arch.get('base_model', 'N/A')}",
        f"- **Pretrained:** {arch.get('pretrained', 'N/A')}",
        f"- **Input size:** {arch.get('input_size', 'N/A')}",
    ]
    for mod in arch.get("modifications", []):
        lines.append(f"- **Modification:** {mod}")
    lines.append("")

    lines += [
        "### Preprocessing Steps",
    ]
    for step in pipeline.get("preprocessing", []):
        lines.append(f"1. {step}")
    lines.append("")

    lines += [
        "### Training Configuration",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Optimizer | {training.get('optimizer', 'N/A')} |",
        f"| Learning rate | {training.get('learning_rate', 'N/A')} |",
        f"| Batch size | {training.get('batch_size', 'N/A')} |",
        f"| Epochs | {training.get('epochs', 'N/A')} |",
        f"| Loss function | {training.get('loss_function', 'N/A')} |",
        "",
        "**Augmentation:** " + ", ".join(training.get("augmentation", [])),
        "",
        "**Regularization:** " + ", ".join(training.get("regularization", [])),
        "",
        "### Evaluation Metrics",
    ]
    for metric in pipeline.get("evaluation", []):
        lines.append(f"- {metric}")
    lines.append("")

    # ── Section 5: Risks ──────────────────────────────────────────
    lines += [
        "## 5. Risks & Tradeoffs",
        "",
        f"**Compute estimate:** {state.get('compute_estimate', 'N/A')}",
        "",
        "| Risk | Severity | Mitigation |",
        "|------|----------|------------|",
    ]
    for r in state.get("risks", []):
        sev = r.get("severity", "medium")
        sev_badge = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}.get(sev, sev)
        lines.append(f"| {r.get('risk')} | {sev_badge} | {r.get('mitigation')} |")
    lines.append("")

    lines += ["### Risk Details", ""]
    for r in state.get("risks", []):
        lines += [
            f"**{r.get('risk')}** ({r.get('severity', 'N/A')} severity)",
            f"> {r.get('description', '')}",
            f"*Mitigation: {r.get('mitigation', '')}*",
            "",
        ]

    # ── Section 6: Next Steps ─────────────────────────────────────
    lines += [
        "## 6. Recommended Next Steps",
        "",
    ]
    for i, step in enumerate(state.get("recommended_next_steps", []), 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    # ── Footer ────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        f"*Steps completed: {', '.join(state.get('steps_completed', []))}*",
    ]
    if state.get("errors"):
        lines.append(f"*Errors: {', '.join(e['step'] for e in state['errors'])}*")

    # ── Write to file ─────────────────────────────────────────────
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Report written to {path}")
    mark_step_done(state, "report_writer")