"""Page 3: Operations Overview."""
from __future__ import annotations

import html
import sqlite3

import pandas as pd
import streamlit as st

from app.charts import (
    chart_completeness_distribution,
    chart_response_time_histogram,
    chart_volume_by_intent,
    chart_weekly_demand,
)
from app.shared_views import OPERATIONAL_URGENCY_NOTE, build_checklist_dataframe, render_priority_section
from app.state import request_nav
from app.ui import message_box, render_page_header
from src.analytics import inbox, insights, kpis
from src.config import Checklist, load_settings, reference_now
from src.database import repository as repo
from src.scoring.completeness import READINESS_NOTE


def _kpi_card_html(label: str, value: str, sub: str) -> str:
    """Build a raw HTML KPI card with all text HTML-escaped."""
    return (
        f'<div class="stq-kpi-card" style="min-height: 11.5rem;">'
        f'<div class="stq-kpi-label">{html.escape(label)}</div>'
        f'<div class="stq-kpi-value">{html.escape(value)}</div>'
        f'<div class="stq-kpi-sub">{html.escape(sub)}</div>'
        f"</div>"
    )


def _render_kpi_cards(K: kpis.KPIs) -> None:
    """Render the six operational KPI summary cards, notes, and calculation definitions."""
    med_str = f"{K.median_hours:.1f} h" if K.median_hours is not None else "n/a"
    p90_str = f"{K.p90_hours:.1f} h" if K.p90_hours is not None else "n/a"
    resp_sub = (
        f"P90: {p90_str} · based on {K.responded} answered inquiries · "
        f"{K.awaiting} open and awaiting a first response · "
        f"{K.closed_without_response} closed without one (neither is included in the median)"
    )

    overdue_sub = f"{K.overdue.fraction_text} follow-up tasks are open and past due"
    comp_val = f"{K.completeness_avg * 100:.1f}%" if K.completeness_avg is not None else "n/a"
    comp_sub = f"n = {K.completeness_n} inquiries · ready for human review, not for binding"

    mature_sub = (
        f"{K.mature.bound.fraction_text} mature quote requests bound · "
        f"{K.mature.quoted.fraction_text} reached a quote"
    )
    if not K.by_intent.empty:
        top_intent = str(K.by_intent.index[0]).replace("_", " ")
        top_count = int(K.by_intent.iloc[0])
        intent_val = f"{len(K.by_intent)} intents"
        intent_sub = f"Top: {top_intent} ({top_count} of {int(K.by_intent.sum())} classified)"
    else:
        intent_val = "0 intents"
        intent_sub = "No classified inquiries in period"

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown(
            _kpi_card_html(
                "Inquiries in period",
                str(K.total),
                f"{K.start:%d %b %Y} to {K.end:%d %b %Y} ({K.responded} answered)",
            ),
            unsafe_allow_html=True,
        )
    with r1c2:
        st.markdown(
            _kpi_card_html("Median first response (hours)", med_str, resp_sub),
            unsafe_allow_html=True,
        )
    with r1c3:
        st.markdown(
            _kpi_card_html("Overdue follow-ups", K.overdue.pct_text, overdue_sub),
            unsafe_allow_html=True,
        )

    st.write("")
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.markdown(
            _kpi_card_html("Average submission completeness", comp_val, comp_sub),
            unsafe_allow_html=True,
        )
    with r2c2:
        st.markdown(
            _kpi_card_html("Quote request → bound (mature cohorts only)", K.mature.bound.pct_text, mature_sub),
            unsafe_allow_html=True,
        )
    with r2c3:
        st.markdown(
            _kpi_card_html("Volume by intent", intent_val, intent_sub),
            unsafe_allow_html=True,
        )

    st.caption(f"**Mature cohort definition:** {K.mature.definition}")
    for note in K.notes:
        st.caption(note)

    with st.expander("How these numbers are calculated"):
        st.markdown(
            "- **Period:** Inquiries whose `received_at` date falls within `[start, end]` (both inclusive).\n"
            "- **First response:** Time from `received_at` to the earliest `first_response` event, computed across "
            "answered inquiries only. Inquiries with no first response are reported separately as open-and-awaiting "
            "versus closed-without-one. P90 uses linear interpolation.\n"
            "- **Overdue follow-up:** Open follow-up tasks whose `due_at` is earlier than the fixed demo clock, "
            "divided by all non-cancelled follow-up tasks for inquiries in the period.\n"
            "- **Completeness:** `provided` fields divided by applicable fields (`not_applicable` excluded), "
            "averaged over inquiries with at least one applicable field. Ready for human review, not a bindable quote.\n"
            f"- **Quote-to-bound (mature cohort):** {K.mature.definition}"
        )


def _render_priority_inbox(conn: sqlite3.Connection, checklist: Checklist) -> None:
    """Render the ranked open-inquiry table, detail inspector, and un-scored review list."""
    now = reference_now()
    ib = inbox.build_inbox(conn, now)

    st.subheader("Priority Inbox (All Open Inquiries)")
    st.caption(
        "The priority inbox shows all currently open inquiries regardless of the date filter above. "
        + OPERATIONAL_URGENCY_NOTE
    )

    if ib.ranked.empty:
        st.info("No scored open inquiries are currently in the inbox.")
    else:
        ranked_ids = [int(x) for x in ib.ranked["Inquiry"].tolist()]
        if st.session_state.get("p3_selected") not in ranked_ids:
            st.session_state["p3_selected"] = ranked_ids[0]

        event = st.dataframe(
            ib.ranked,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="p3_inbox_df",
        )
        sel_rows = []
        if event is not None and hasattr(event, "selection"):
            sel_rows = list(getattr(event.selection, "rows", []) or [])
        elif isinstance(event, dict):
            sel_rows = list(event.get("selection", {}).get("rows", []) or [])

        if sel_rows and sel_rows[0] != st.session_state.get("_p3_last_df_row"):
            picked_row = int(sel_rows[0])
            if 0 <= picked_row < len(ranked_ids):
                st.session_state["_p3_last_df_row"] = picked_row
                st.session_state["p3_selected"] = ranked_ids[picked_row]

        by_id_row = {int(r["Inquiry"]): r for _, r in ib.ranked.iterrows()}

        def _fmt_inbox_choice(iid: int) -> str:
            r = by_id_row[iid]
            return f"Rank #{r['Rank']} · Inquiry #{iid} · Score {r['Score']} · {r['Intent']} — {r['Message'][:55]}"

        chosen_id = st.selectbox(
            "Or pick an inquiry to inspect",
            options=ranked_ids,
            format_func=_fmt_inbox_choice,
            key="p3_selected",
        )
        analysis = ib.analyses.get(int(chosen_id))
        if analysis is not None:
            with st.container(border=True):
                st.markdown(f"#### Inquiry #{chosen_id} Breakdown")
                render_priority_section(analysis)
                st.markdown("**Customer message**")
                message_box(analysis.inquiry["message"])
                st.markdown("**Checklist statuses**")
                c_rows = repo.checklist_rows(conn, int(chosen_id))
                st.dataframe(build_checklist_dataframe(checklist, c_rows), width="stretch", hide_index=True)
                st.caption(READINESS_NOTE)
                st.caption(checklist.disclaimer)
                if st.button("Open in Submission Review →", key="btn_p3_open_review", type="primary"):
                    request_nav("Submission Review", inquiry_id=int(chosen_id))

    with st.expander(
        f"Needs human review first: {len(ib.review)} open inquiries without a score",
        expanded=False,
    ):
        if ib.review.empty:
            st.caption("No open inquiries are currently awaiting human review without a score.")
        else:
            st.dataframe(ib.review, width="stretch", hide_index=True)


def _render_overdue_table(K: kpis.KPIs) -> None:
    """Render the table of open and past-due follow-up tasks, most overdue first."""
    st.subheader("Overdue Follow-ups")
    if K.overdue_tasks.empty:
        st.caption("No overdue follow-up tasks for inquiries received in the selected period.")
        return

    rows = []
    for _, r in K.overdue_tasks.iterrows():
        due_str = pd.Timestamp(r["due_at"]).strftime("%Y-%m-%d %H:%M")
        rows.append(
            {
                "Inquiry": int(r["inquiry_id"]),
                "Task": str(r["task_type"]).replace("_", " "),
                "Was due": due_str,
                "Days overdue": round(float(r["days_overdue"]), 1),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_insights(frames: kpis.Frames, K: kpis.KPIs) -> None:
    """Render the four descriptive, hypothesis-labelled operational insights."""
    st.subheader("Operational Insights")
    st.caption(
        "Rule-based and descriptive only: no causal claims and no revenue estimates. "
        "Because this dataset is synthetic, any pattern may reflect how the data was generated."
    )
    now = reference_now()
    for ins in insights.build_insights(frames, K, now):
        with st.expander(ins.title, expanded=True):
            st.markdown(f"**Observation.** {ins.observation}")
            st.markdown("**Possible explanations (hypotheses, not findings)**")
            for hyp in ins.hypotheses:
                st.markdown(f"- {hyp}")
            st.markdown(f"**Suggested action.** {ins.action}")
            st.markdown(f"**Metric to monitor.** {ins.metric}")


def render_ops_page(conn: sqlite3.Connection, checklist: Checklist) -> None:
    """Render Page 3: Operations Overview."""
    render_page_header(
        "Operations Overview",
        "Response-time KPIs, the ranked priority inbox, overdue follow-ups, and descriptive operational insights.",
    )

    all_frames = kpis.load_frames(conn)
    if all_frames.inquiries.empty:
        st.info("No inquiries are stored in the database yet.")
        return

    min_d, max_d = kpis.date_bounds(all_frames)
    date_sel = st.date_input(
        "Inquiry received date range (inclusive)",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key="ops_date_range",
    )
    if not isinstance(date_sel, (tuple, list)) or len(date_sel) != 2 or date_sel[0] is None or date_sel[1] is None:
        st.info("Pick an end date to complete the period selection.")
        return

    start_d, end_d = min(date_sel[0], date_sel[1]), max(date_sel[0], date_sel[1])
    frames = kpis.filter_period(all_frames, start_d, end_d)
    now = reference_now()
    mature_days = int(load_settings()["mature_cohort_days"])
    K = kpis.compute_kpis(frames, now, mature_days, start_d, end_d)

    _render_kpi_cards(K)
    st.divider()

    st.subheader("Operational Charts")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_volume_by_intent(K.by_intent), width="stretch")
    with c2:
        st.plotly_chart(
            chart_response_time_histogram(K.response_hours, K.median_hours, K.p90_hours),
            width="stretch",
        )
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_weekly_demand(frames.inquiries), width="stretch")
    with c4:
        st.plotly_chart(
            chart_completeness_distribution(kpis.completeness_by_inquiry(frames)),
            width="stretch",
        )

    st.divider()
    _render_priority_inbox(conn, checklist)
    st.divider()
    _render_overdue_table(K)
    st.divider()
    _render_insights(frames, K)
