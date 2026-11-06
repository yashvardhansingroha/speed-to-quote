"""Shared rendering helpers used across Inquiry Triage, Submission Review, and Operations Overview."""
from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd
import streamlit as st

from app.ui import md
from src.analysis import Analysis
from src.config import Checklist
from src.extraction.llm_client import LLMClient
from src.extraction.schema import NOT_PROVIDED

STATUS_ICONS: dict[str, str] = {
    "provided": "✓",
    "missing": "✗",
    "unclear": "?",
    "not_applicable": "—",
}

OPERATIONAL_URGENCY_NOTE = (
    "Priority score measures operational urgency (how soon a person should look at this inquiry). "
    "It is not a prediction of conversion, deal size, or customer quality."
)

QUOTE_BACKING_NOTE = (
    "Every value must be backed by a quote copied exactly from the message, or `not_provided`; "
    "quotes are verified by code."
)


def format_status_badge(status: str) -> str:
    """Return an icon-prefixed checklist status label."""
    icon = STATUS_ICONS.get(status, "•")
    return f"{icon} {status}"


def quote_check_text(verified: Optional[int | bool]) -> str:
    """Return human-readable verification status for an evidence quote."""
    if verified is None:
        return "n.a."
    return "found" if bool(verified) else "NOT found"


def render_engine_summary(analysis: Analysis, client: LLMClient) -> None:
    """Show inquiry metadata, extraction engine used, and any fallback or offline notice."""
    inq = analysis.inquiry
    received = str(inq["received_at"]).replace("T", " ")[:16]
    engine_label = (
        f"LLM ({analysis.model or 'configured model'})"
        if analysis.method == "llm"
        else "rules-based baseline"
    )
    fallback_part = f" · Fallback reason: {md(analysis.fallback_reason)}" if analysis.fallback_reason else ""
    st.caption(
        f"Inquiry **#{inq['id']}** · Received: **{md(received)}** · Status: **{md(inq['status'])}** · "
        f"Channel: **{md(inq['channel'])}** · Engine used: **{md(engine_label)}**{fallback_part}"
    )

    reason = analysis.fallback_reason or ""
    if analysis.method == "rules" and reason.startswith(("LLMCallError", "InvalidLLMResponse", "unexpected")):
        st.warning(
            f"The LLM extraction step failed ({md(reason)}), so the deterministic rules-based baseline was used instead."
        )
    elif analysis.method == "rules" and client.available and reason not in ("message_too_short", "message_too_long"):
        st.info(
            "This record was extracted with the rules-based baseline. Click re-run extraction to analyse it with the configured LLM."
        )


def format_deadline_display(analysis: Analysis) -> tuple[str, str]:
    """Return (main_text, sub_text) for the Deadline summary card."""
    tri = analysis.triage
    ex = analysis.extraction
    raw_words = ex.deadline.value or (
        ex.deadline.evidence if ex.deadline.evidence != NOT_PROVIDED else None
    )
    words_caption = f'Customer wording: "{raw_words}"' if raw_words else "No deadline wording quoted"

    if tri.deadline_date is None:
        return "None stated", "No actionable future or recent deadline found"
    if tri.deadline_approximate and tri.deadline_latest is not None:
        span = f"{tri.deadline_date:%b %d} – {tri.deadline_latest:%b %d} (approx.)"
        return span, words_caption
    return f"{tri.deadline_date:%b %d, %Y}", words_caption


def render_priority_section(analysis: Analysis) -> None:
    """Render either the human-review routing box (no score) or the 0-100 score and factor breakdown."""
    tri = analysis.triage
    st.subheader("Operational Priority")
    st.caption(OPERATIONAL_URGENCY_NOTE)

    if tri.needs_review or tri.priority is None:
        reasons_md = "\n".join(f"- {md(r)}" for r in tri.review_reasons)
        st.error(f"**Routed to human review: not scored**\n\n{reasons_md}")
    else:
        score = tri.priority.score
        col_score, col_bar = st.columns([1, 3])
        with col_score:
            st.metric("Priority score", f"{score} / 100")
        with col_bar:
            st.progress(min(max(score / 100.0, 0.0), 1.0))
            st.caption("Weighted sum of urgency (max 25), deadline (max 40), waiting time (max 20), and follow-up state (max 15).")

        factor_rows = [
            {
                "Factor": f.name.capitalize(),
                "Points": f"{f.points} / {f.max_points}",
                "Why": f.reason,
            }
            for f in tri.priority.factors
        ]
        st.dataframe(pd.DataFrame(factor_rows), width="stretch", hide_index=True)

    for hint in tri.hints:
        st.warning(md(hint))


def build_checklist_dataframe(checklist: Checklist, rows: dict[str, sqlite3.Row]) -> pd.DataFrame:
    """Build the 12-field checklist summary table including any manual human corrections."""
    table_rows = []
    for f in checklist.fields:
        r = rows.get(f.key)
        if r is None:
            table_rows.append(
                {
                    "Field": f.label,
                    "Status": format_status_badge("missing"),
                    "Value": "—",
                    "Evidence quote": NOT_PROVIDED,
                    "Quote check": "n.a.",
                    "Source": "—",
                }
            )
            continue
        table_rows.append(
            {
                "Field": f.label,
                "Status": format_status_badge(str(r["status"])),
                "Value": r["value"] if r["value"] else "—",
                "Evidence quote": r["evidence"] if r["evidence"] else NOT_PROVIDED,
                "Quote check": quote_check_text(r["evidence_verified"]),
                "Source": str(r["source"]),
            }
        )
    return pd.DataFrame(table_rows)
