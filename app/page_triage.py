"""Page 1: Inquiry Triage."""
from __future__ import annotations

import html
import sqlite3
from typing import Optional

import streamlit as st

from app.shared_views import (
    QUOTE_BACKING_NOTE,
    build_checklist_dataframe,
    format_deadline_display,
    quote_check_text,
    render_engine_summary,
    render_priority_section,
)
from app.state import request_nav, select_inquiry_widget
from app.ui import md, message_box, render_flash, render_mode_banner, render_page_header, set_flash
from src.analysis import Analysis, analyze_inquiry, load_analysis
from src.config import Checklist, load_settings, reference_now
from src.database import repository as repo
from src.extraction.llm_client import LLMClient
from src.scoring.completeness import READINESS_NOTE


def _render_triage_cards(analysis: Analysis) -> None:
    """Render the Intent, Stated urgency, and Deadline summary cards."""
    ex = analysis.extraction
    deadline_main, deadline_sub = format_deadline_display(analysis)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stq-kpi-card">'
            f'<div class="stq-kpi-label">Intent</div>'
            f'<div class="stq-kpi-value">{html.escape(str(ex.intent))}</div>'
            f'<div class="stq-kpi-sub">Quote check: {html.escape(str(quote_check_text(ex.intent_evidence_verified)))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stq-kpi-card">'
            f'<div class="stq-kpi-label">Stated urgency</div>'
            f'<div class="stq-kpi-value">{html.escape(str(ex.urgency))}</div>'
            f'<div class="stq-kpi-sub">Quote check: {html.escape(str(quote_check_text(ex.urgency_evidence_verified)))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stq-kpi-card">'
            f'<div class="stq-kpi-label">Deadline</div>'
            f'<div class="stq-kpi-value">{html.escape(str(deadline_main))}</div>'
            f'<div class="stq-kpi-sub">{html.escape(str(deadline_sub))}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if ex.intent_note:
        st.warning(f"Intent note: {md(ex.intent_note)}")


def _render_supporting_quotes(analysis: Analysis) -> None:
    """Render the expander showing supporting quotes for intent, urgency, and business description."""
    ex = analysis.extraction
    bd = ex.business_description
    with st.expander("Supporting quotes"):
        st.markdown(
            f"- **Intent (`{md(ex.intent)}`):** `{md(ex.intent_evidence)}` "
            f"— Quote check: **{quote_check_text(ex.intent_evidence_verified)}**"
        )
        st.markdown(
            f"- **Stated urgency (`{md(ex.urgency)}`):** `{md(ex.urgency_evidence)}` "
            f"— Quote check: **{quote_check_text(ex.urgency_evidence_verified)}**"
        )
        bd_val = md(bd.value) if bd.value else "—"
        st.markdown(
            f"- **Business description (`{md(bd.status)}` · value: {bd_val}):** `{md(bd.evidence)}` "
            f"— Quote check: **{quote_check_text(bd.evidence_verified)}**"
        )
        if bd.note:
            st.caption(f"Business description note: {md(bd.note)}")


def _render_analysis_result(
    conn: sqlite3.Connection,
    checklist: Checklist,
    client: LLMClient,
    analysis: Analysis,
) -> None:
    """Render the full extraction, triage priority, quotes, checklist table, and review navigation button."""
    inq_id = int(analysis.inquiry["id"])
    st.subheader("Customer Message")
    message_box(analysis.inquiry["message"])
    render_engine_summary(analysis, client)

    render_priority_section(analysis)
    st.divider()
    _render_triage_cards(analysis)
    _render_supporting_quotes(analysis)

    st.subheader("Submission Checklist")
    rows = repo.checklist_rows(conn, inq_id)
    df = build_checklist_dataframe(checklist, rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(QUOTE_BACKING_NOTE)
    st.caption(md(checklist.disclaimer))
    st.caption(READINESS_NOTE)

    if st.button("Open in Submission Review →", key="btn_open_in_review", type="primary"):
        request_nav("Submission Review", inquiry_id=inq_id)


def render_triage_page(conn: sqlite3.Connection, checklist: Checklist, client: LLMClient) -> None:
    """Render Page 1: Inquiry Triage."""
    render_page_header(
        "Inquiry Triage",
        "Turn an unstructured trucking inquiry into a structured extraction and explainable operational priority score.",
    )
    render_mode_banner(client)
    render_flash("triage")

    now = reference_now()
    source = st.radio(
        "Inquiry source",
        options=["Select from inbox", "Paste a message"],
        horizontal=True,
        key="triage_source",
    )

    if source == "Select from inbox":
        inquiry_id = select_inquiry_widget(conn, "triage_inquiry_select", "Select an inquiry from the inbox")
        if inquiry_id is None:
            st.info("No inquiries are stored in the database yet. Switch to 'Paste a message' to add one.")
            return

        if st.button("Re-run extraction now (uses the LLM if available)", key="btn_rerun_extraction"):
            analyze_inquiry(conn, inquiry_id, checklist, client, now)
            set_flash(
                "success",
                "Extraction re-run complete. Any existing human checklist corrections were preserved.",
                page_key="triage",
            )
            st.rerun()

        analysis = load_analysis(conn, inquiry_id, now)
        if analysis is None:
            st.info("This inquiry has not been analysed yet.")
            if st.button("Run extraction now", key="btn_initial_extract_triage"):
                analyze_inquiry(conn, inquiry_id, checklist, client, now)
                st.rerun()
            return

        _render_analysis_result(conn, checklist, client, analysis)
        return

    # Paste mode
    max_chars = int(load_settings()["max_message_chars"])
    pasted_text = st.text_area(
        "Paste an unstructured customer inquiry",
        height=150,
        max_chars=max_chars,
        key="paste_message_text",
        placeholder="Example: Hi, we run 12 trucks in Texas and Oklahoma. Our insurance expires next month...",
    )
    st.caption(
        "Pasted messages are saved to the demo SQLite database with the fixed demo timestamp so they can be reviewed."
    )
    st.warning(
        "**Do not paste real customer or personal information.** The demo database is shared, and when an API key "
        "is configured the message text is sent to the LLM provider."
    )

    if st.button("Analyze", key="btn_analyze_paste", type="primary"):
        last_text = st.session_state.get("last_pasted_text")
        last_id: Optional[int] = st.session_state.get("last_pasted_inquiry_id")
        if last_text == pasted_text and last_id is not None and repo.get_inquiry(conn, last_id) is not None:
            inquiry_id = int(last_id)
        else:
            inquiry_id = repo.create_pasted_inquiry(conn, pasted_text, now)
            st.session_state["last_pasted_text"] = pasted_text
            st.session_state["last_pasted_inquiry_id"] = inquiry_id

        analyze_inquiry(conn, inquiry_id, checklist, client, now)
        st.session_state["current_inquiry_id"] = inquiry_id

    active_pasted_id: Optional[int] = st.session_state.get("last_pasted_inquiry_id")
    if active_pasted_id is not None:
        analysis = load_analysis(conn, int(active_pasted_id), now)
        if analysis is not None:
            st.divider()
            _render_analysis_result(conn, checklist, client, analysis)
