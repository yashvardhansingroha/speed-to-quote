"""Try the triage pipeline from the terminal.

    python -m src.demo_cli "Hi, we run 12 trucks in Texas and Oklahoma. Our insurance expires next month..."

Uses the LLM if a key is configured (.env or environment), otherwise the offline rules baseline.
"""
from __future__ import annotations

import sys

from src.config import load_checklist, reference_now
from src.extraction.llm_client import LLMClient, load_llm_config
from src.extraction.pipeline import run_extraction
from src.scoring.triage import triage_inquiry


def main(text: str) -> None:
    checklist, now = load_checklist(), reference_now()
    client = LLMClient(load_llm_config())
    outcome = run_extraction(text, checklist, now, client)
    ex = outcome.extraction

    banner = "Offline mode: rules-based" if outcome.offline else f"LLM: {outcome.model}"
    print(f"[{banner}]  fallback_reason={outcome.fallback_reason}  latency={outcome.latency_ms:.0f}ms")
    print(f"intent={ex.intent}  urgency={ex.urgency}")
    print(f"deadline: {ex.deadline.value!r}  evidence: {ex.deadline.evidence!r}")
    print("\nChecklist:")
    for f in checklist.fields:
        r = ex.fields[f.key]
        print(f"  {r.status:<14} {f.label:<38} value={r.value!r}  evidence={r.evidence!r}")

    tri = triage_inquiry(text, ex, now, now)
    print()
    if tri.needs_review:
        print("ROUTED TO HUMAN REVIEW (not scored):", *tri.review_reasons, sep="\n  - ")
    else:
        print(f"Priority score: {tri.priority.score}/100   (operational urgency, not a prediction of conversion)")
        for reason in tri.priority.explain():
            print("  ", reason)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(" ".join(sys.argv[1:]))
