"""Page 2: Submission Review."""
from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from app.shared_views import STATUS_ICONS, quote_check_text, render_engine_summary
from app.state import select_inquiry_widget
from app.ui import md, message_box, render_flash, render_mode_banner, render_page_header, set_flash
from src.analysis import Analysis, analyze_inquiry, draft_context, load_analysis
from src.config import Checklist, reference_now
from src.database import repository as repo
from src.drafting import workflow
from src.drafting.draft import approval_blockers, find_violations, generate_draft
from src.extraction.llm_client import LLMClient
from src.extraction.schema import NOT_PROVIDED
from src.scoring.completeness import READINESS_NOTE, compute_completeness


def _render_completeness_block(checklist: Checklist, rows: dict[str, sqlite3.Row]) -> None:
    """Render the submission completeness summary, readiness note, and field status lists."""
    statuses = {f.key: str(rows[f.key]["status"]) for f in checklist.fields if f.key in rows}
    comp = compute_completeness(statuses)

    st.subheader("Submission Completeness")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Applicable fields supplied", f"{comp.provided} of {comp.applicable}")
    with c2:
        st.progress(min(max(comp.fraction, 0.0), 1.0))
        st.markdown(f"**{comp.text}**")
        st.caption(READINESS_NOTE)

    missing_labels = [checklist.get(k).label for k in comp.missing]
    unclear_labels = [checklist.get(k).label for k in comp.unclear]
    na_labels = [checklist.get(k).label for k in comp.not_applicable]

    col_m, col_u, col_na = st.columns(3)
    with col_m:
        st.markdown(f"**Missing ({len(missing_labels)}):** " + (md(", ".join(missing_labels)) if missing_labels else "None"))
    with col_u:
        st.markdown(f"**Unclear ({len(unclear_labels)}):** " + (md(", ".join(unclear_labels)) if unclear_labels else "None"))
    with col_na:
        st.markdown(
            f"**Not applicable ({len(na_labels)}):** " + (md(", ".join(na_labels)) if na_labels else "None")
        )
    st.caption(md(checklist.disclaimer))


def _render_checklist_corrections(
    conn: sqlite3.Connection,
    inquiry_id: int,
    checklist: Checklist,
    analysis: Analysis,
    rows: dict[str, sqlite3.Row],
) -> None:
    """Render one expander per checklist field with its details and human correction form."""
    now = reference_now()
    st.subheader("Checklist & Human Corrections")
    st.caption("Expand any field to inspect its supporting quote or record a manual correction in the audit trail.")

    for f in checklist.fields:
        r = rows.get(f.key)
        if r is None:
            continue
        status = str(r["status"])
        val = r["value"] or ""
        icon = STATUS_ICONS.get(status, "•")
        val_preview = val[:40] if val else "—"
        title = f"{icon} {md(f.label)} · {md(status)} · {md(val_preview)}"

        with st.expander(title):
            col_info, col_form = st.columns([3, 2])
            with col_info:
                st.markdown(f"**Status:** `{md(status)}` · **Source:** `{md(r['source'])}`")
                st.markdown(f"**Value:** {md(val) if val else '—'}")
                ev = r["evidence"] or NOT_PROVIDED
                if ev == NOT_PROVIDED:
                    st.markdown("**Evidence quote:** `not_provided`: the message gives nothing to quote")
                else:
                    st.markdown(f"**Evidence quote:** `{md(ev)}`")
                st.markdown(f"**Quote verified:** {md(quote_check_text(r['evidence_verified']))}")

                ex_field = analysis.extraction.fields.get(f.key)
                if ex_field and ex_field.note:
                    st.caption(f"Extraction note: {md(ex_field.note)}")
                if r["source"] == "manual":
                    ts_str = str(r["updated_at"]).replace("T", " ")[:19]
                    st.info(f"Corrected by a human at {md(ts_str)}")
                st.caption(md(f.description))

            with col_form:
                with st.form(key=f"fix_{inquiry_id}_{f.key}"):
                    idx = list(repo.VALID_STATUSES).index(status) if status in repo.VALID_STATUSES else 0
                    new_status = st.selectbox(
                        "Correct status",
                        options=list(repo.VALID_STATUSES),
                        index=idx,
                        key=f"fix_status_{inquiry_id}_{f.key}",
                    )
                    new_val = st.text_input(
                        "Correct value",
                        value=val,
                        key=f"fix_value_{inquiry_id}_{f.key}",
                    )
                    new_reason = st.text_input(
                        "Reason (optional)",
                        value="",
                        key=f"fix_reason_{inquiry_id}_{f.key}",
                    )
                    submitted = st.form_submit_button("Save correction")
                    if submitted:
                        try:
                            repo.save_correction(
                                conn,
                                inquiry_id,
                                f.key,
                                new_status,
                                new_val,
                                now,
                                new_reason,
                            )
                            set_flash("success", f"Saved manual correction for {f.label}.", page_key="review")
                            st.rerun()
                        except ValueError as exc:
                            set_flash("error", str(exc), page_key="review")
                            st.rerun()


def _render_draft_section(
    conn: sqlite3.Connection,
    inquiry_id: int,
    checklist: Checklist,
    analysis: Analysis,
    client: LLMClient,
) -> None:
    """Render follow-up draft generation, safety scan feedback, and Save / Approve / Reject workflow."""
    now = reference_now()
    ctx = draft_context(conn, analysis, checklist)
    requested = ctx.requested()
    fp = workflow.checklist_fingerprint(ctx.statuses)

    st.subheader("Follow-up Draft")
    st.info(
        '**Human approval required: nothing is sent.** This app never sends emails or messages; '
        '"Approve" only records that a person signed the text off.'
    )
    st.caption(
        f"The draft asks only about the **{len(requested)}** missing or unclear checklist field(s) and never mentions "
        "premiums, coverage promises, eligibility, or deadlines the customer did not state."
    )

    draft = workflow.get_current_draft(conn, inquiry_id)
    if draft is None:
        if st.button("Generate follow-up draft", key=f"btn_gen_draft_{inquiry_id}", type="primary"):
            res = generate_draft(ctx, client)
            workflow.create_draft(
                conn, inquiry_id, res.body, res.generated_by, now, note=res.note, checklist_hash=fp
            )
            set_flash("success", "Follow-up draft generated.", page_key="review")
            st.rerun()
        return

    draft_id = int(draft["id"])
    status = str(draft["status"])
    stale = workflow.is_stale(draft, ctx.statuses)
    status_label = status.capitalize() + (" · Outdated" if stale else "")
    created_str = str(draft["created_at"]).replace("T", " ")[:19]

    st.markdown(
        f"**Draft status:** `{md(status_label)}` · **Generated by:** `{md(draft['generated_by'])}` · "
        f"**Created:** `{md(created_str)}`"
    )
    if draft["note"]:
        st.caption(f"Draft note: {md(draft['note'])}")

    if stale:
        st.warning(
            "**Outdated:** The checklist has changed since this draft was written, so it cannot be approved. "
            "Generate a new draft for the current checklist."
        )

    if status == "draft":
        text_key = f"draft_text_{draft_id}"
        if text_key not in st.session_state:
            st.session_state[text_key] = str(draft["body"])
        current_text = st.text_area("Edit follow-up draft", height=240, key=text_key)

        provided_labels = [f.label for f in ctx.provided_fields()]
        problems = find_violations(current_text, ctx.message, provided_labels)
        blockers = approval_blockers(problems)
        advisories = [p for p in problems if p not in blockers]
        is_empty = not (current_text or "").strip()

        if is_empty:
            st.error("The draft text is empty and cannot be approved or saved.")
        if blockers:
            joined_b = "; ".join(blockers)
            st.error(
                f"Blocking content rule violation(s): {md(joined_b)}. "
                "This draft cannot be approved until fixed, though it can still be saved or rejected."
            )
        if advisories:
            st.warning("Advisory reminder(s): " + md("; ".join(advisories)))

        reason_val = st.text_input(
            "Reason (optional, recorded when rejecting)",
            value="",
            key=f"draft_reason_{draft_id}",
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button(
                "Approve",
                key=f"btn_approve_{draft_id}",
                type="primary",
                disabled=bool(stale or blockers or is_empty),
            ):
                try:
                    workflow.approve_draft(conn, draft_id, current_text, now, ctx)
                    set_flash("success", "Draft approved (nothing was sent).", page_key="review")
                    st.rerun()
                except workflow.DraftError as exc:
                    set_flash("error", str(exc), page_key="review")
                    st.rerun()
        with c2:
            if st.button("Reject", key=f"btn_reject_{draft_id}"):
                try:
                    workflow.reject_draft(conn, draft_id, current_text, now, reason=reason_val)
                    set_flash("warning", "Draft rejected.", page_key="review")
                    st.rerun()
                except workflow.DraftError as exc:
                    set_flash("error", str(exc), page_key="review")
                    st.rerun()
        with c3:
            if st.button("Save draft", key=f"btn_save_{draft_id}"):
                try:
                    workflow.save_draft(conn, draft_id, current_text, now)
                    set_flash("success", "Draft saved.", page_key="review")
                    st.rerun()
                except workflow.DraftError as exc:
                    set_flash("error", str(exc), page_key="review")
                    st.rerun()
        with c4:
            if st.button("Discard and generate a new draft", key=f"btn_regen_{draft_id}"):
                res = generate_draft(ctx, client)
                workflow.create_draft(
                    conn, inquiry_id, res.body, res.generated_by, now, note=res.note, checklist_hash=fp
                )
                set_flash("success", "Generated a new follow-up draft.", page_key="review")
                st.rerun()
    else:
        if status == "approved" and stale:
            st.info(
                "This approved draft is now historical because the checklist changed after sign-off. "
                "Generate a new draft for the updated checklist."
            )
        message_box(str(draft["body"]))
        if status == "rejected":
            history = workflow.audit_history(conn, inquiry_id)
            rej_reasons = [r["reason"] for r in history if r["action"] == "reject" and r["reason"]]
            if rej_reasons:
                st.caption(f"Rejection reason: {md(rej_reasons[-1])}")

        if st.button("Generate a new draft", key=f"btn_new_draft_{draft_id}"):
            res = generate_draft(ctx, client)
            workflow.create_draft(
                conn, inquiry_id, res.body, res.generated_by, now, note=res.note, checklist_hash=fp
            )
            set_flash("success", "Generated a new follow-up draft.", page_key="review")
            st.rerun()


def _render_audit_history(conn: sqlite3.Connection, inquiry_id: int) -> None:
    """Render the audit log table and full-text expander for this inquiry."""
    st.subheader("Audit History")
    audit_rows = workflow.audit_history(conn, inquiry_id)
    if not audit_rows:
        st.caption("No audit actions recorded for this inquiry yet.")
        return

    table_data = []
    draft_actions = []
    for r in audit_rows:
        orig = r["original_draft"]
        edit = r["edited_draft"]
        if orig is not None and edit is not None:
            changed = "Yes" if orig != edit else "No"
        else:
            changed = "—"
        table_data.append(
            {
                "Time": str(r["ts"]).replace("T", " ")[:19],
                "Action": str(r["action"]),
                "Reason / field": r["reason"] or "—",
                "Original": (orig.replace("\n", " ")[:60] if orig else "—"),
                "Edited": (edit.replace("\n", " ")[:60] if edit else "—"),
                "Changed by human": changed,
            }
        )
        if r["action"] in ("save_draft", "approve", "reject"):
            draft_actions.append(r)

    st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

    if draft_actions:
        with st.expander("Full text of save / approve / reject actions"):
            for r in draft_actions:
                st.markdown(
                    f"**Action:** `{md(r['action'])}` at `{md(str(r['ts']).replace('T', ' ')[:19])}`"
                    + (f" · **Reason:** {md(r['reason'])}" if r["reason"] else "")
                )
                c_orig, c_edit = st.columns(2)
                with c_orig:
                    st.caption("Original generated draft")
                    message_box(r["original_draft"] or "")
                with c_edit:
                    st.caption("Text at time of action")
                    message_box(r["edited_draft"] or "")


def render_review_page(conn: sqlite3.Connection, checklist: Checklist, client: LLMClient) -> None:
    """Render Page 2: Submission Review."""
    render_page_header(
        "Submission Review",
        "Review checklist completeness, record human corrections, and approve or reject a follow-up draft.",
    )
    render_mode_banner(client)
    render_flash("review")

    now = reference_now()
    inquiry_id = select_inquiry_widget(conn, "review_inquiry_select", "Select an inquiry")
    if inquiry_id is None:
        st.info("No inquiries are stored in the database yet.")
        return

    analysis = load_analysis(conn, inquiry_id, now)
    if analysis is None:
        st.info("This inquiry has not been analysed yet.")
        if st.button("Run extraction", key="btn_run_extraction_review", type="primary"):
            analyze_inquiry(conn, inquiry_id, checklist, client, now)
            st.rerun()
        return

    with st.expander("Customer message", expanded=False):
        message_box(analysis.inquiry["message"])
    render_engine_summary(analysis, client)

    rows = repo.checklist_rows(conn, inquiry_id)
    _render_completeness_block(checklist, rows)
    st.divider()
    _render_checklist_corrections(conn, inquiry_id, checklist, analysis, rows)
    st.divider()
    _render_draft_section(conn, inquiry_id, checklist, analysis, client)
    st.divider()
    _render_audit_history(conn, inquiry_id)
