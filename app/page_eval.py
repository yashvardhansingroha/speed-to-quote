"""Page 4: Model Evaluation (reads evaluation files only; never calls the LLM API or writes labels)."""
from __future__ import annotations

import html
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui import md, message_box, render_page_header
from src.config import Checklist
from src.evaluation.export import unlabeled_path
from src.evaluation.failures import FailureCase
from src.evaluation.report import EvalReport, SystemMetrics, build_report


def _eval_card_html(label: str, value: str, sub: str) -> str:
    """Build a raw HTML status/metric card with HTML-escaped content."""
    return (
        f'<div class="stq-kpi-card" style="min-height: 9.5rem;">'
        f'<div class="stq-kpi-label">{html.escape(label)}</div>'
        f'<div class="stq-kpi-value">{html.escape(value)}</div>'
        f'<div class="stq-kpi-sub">{html.escape(sub)}</div>'
        f"</div>"
    )


def _fmt_pct(val: Optional[float]) -> str:
    return f"{val * 100:.1f}%" if val is not None else "n/a"


def _fmt_ms(val: Optional[float]) -> str:
    if val is None:
        return "n/a"
    if val >= 100.0:
        return f"{val:.0f} ms ({val / 1000.0:.2f} s)"
    return f"{val:.2f} ms"


def _chart_confusion_matrix(cm: pd.DataFrame, title: str) -> go.Figure:
    """Build a single-axis heatmap for an intent confusion matrix (rows = labelled, columns = predicted)."""
    x_labels = [str(c).replace("_", " ") for c in cm.columns]
    y_labels = [str(r).replace("_", " ") for r in cm.index]
    z_vals = cm.values.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            showscale=False,
            text=z_vals,
            texttemplate="%{text}",
            hovertemplate="Labelled: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 15}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 16, "r": 16, "t": 44, "b": 44},
        height=340,
        xaxis={"title": "Predicted intent", "tickangle": -30},
        yaxis={"title": "Labelled (true) intent", "autorange": "reversed"},
    )
    return fig


def _render_status_row(report: EvalReport) -> None:
    """Render the 3 status cards for the evaluation set, manual labels, and LLM cache."""
    if report.exported:
        exp_val = f"{report.n_messages} messages exported"
        exp_sub = "eval/eval_unlabeled.csv (separate seed from operational data)"
    else:
        exp_val = "Not exported yet"
        exp_sub = "Run python -m src.evaluation.export to create eval/eval_unlabeled.csv"

    lbl_val = "Ready" if report.label_state.ready else "Awaiting labels"
    lbl_sub = report.label_state.message

    stats = report.llm_stats
    if stats is not None:
        cached_done = stats.n_llm_path - stats.n_missing
        cch_val = f"{cached_done} of {stats.n_llm_path} LLM-path messages"
        cch_sub = (
            f"model {report.llm_model or 'unknown'} · "
            f"{stats.n_skipped_short} too short: answered by rules, as in the app"
        )
    else:
        cch_val = "No cache loaded"
        cch_sub = "eval/llm_cache.json not found or empty"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_eval_card_html("Evaluation set", exp_val, exp_sub), unsafe_allow_html=True)
    with c2:
        st.markdown(_eval_card_html("Manual labels", lbl_val, lbl_sub), unsafe_allow_html=True)
    with c3:
        st.markdown(_eval_card_html("LLM cache", cch_val, cch_sub), unsafe_allow_html=True)


def _render_cached_run_measurements(report: EvalReport) -> None:
    """Render schema-validity, API-error, and latency measurements that do not require human labels."""
    st.subheader("Measured from the cached run (no labels needed)")
    stats = report.llm_stats
    rules_lat = f"rules alone: {_fmt_ms(report.rules_latency_ms)}"

    if stats is None:
        st.info(f"No cached LLM run found in `eval/llm_cache.json` ({rules_lat}).")
        return

    inv_rate = _fmt_pct(stats.invalid_rate)
    inv_sub = f"{stats.n_invalid} of {stats.n_responses} responses failed JSON or schema validation"
    err_val = f"{stats.n_api_error} of {stats.n_llm_path} calls"
    err_sub = f"{stats.n_skipped_short} short message(s) routed directly to rules without calling the API"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_eval_card_html("Invalid / malformed responses", inv_rate, inv_sub), unsafe_allow_html=True)
    with c2:
        st.markdown(_eval_card_html("API errors", err_val, err_sub), unsafe_allow_html=True)
    with c3:
        st.markdown(
            _eval_card_html(
                "Median API-call latency",
                _fmt_ms(report.llm_api_latency_ms),
                "Historical cached API round-trip (LLM-path calls only)",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _eval_card_html(
                "Median full-pipeline latency",
                _fmt_ms(report.llm_pipeline_latency_ms),
                f"API call + validation + quote checks ({rules_lat})",
            ),
            unsafe_allow_html=True,
        )

    st.caption(
        "**Important:** 0 malformed responses only means the output matched the JSON schema. "
        "It says nothing about whether the extracted values are correct."
    )


def _render_labeled_comparison(report: EvalReport, checklist: Checklist) -> None:
    """Render the LLM vs Rules comparison table, confusion matrices, per-field breakdown, and failures."""
    assert report.rules is not None
    rules_m: SystemMetrics = report.rules
    llm_m: Optional[SystemMetrics] = report.llm
    stats = report.llm_stats

    st.subheader("Pipeline Comparison (Human-Labelled Set)")
    if llm_m is None:
        st.info(
            "The LLM cache is missing or incomplete, so only Rules baseline metrics are shown below. "
            "Run `python -m src.evaluation.run_eval` to populate `eval/llm_cache.json`."
        )

    def _llm_col(fn, fallback: str = "n/a (cache incomplete)") -> str:
        return fn(llm_m) if llm_m is not None else fallback

    inv_str = (
        f"{_fmt_pct(stats.invalid_rate)} ({stats.n_invalid} of {stats.n_responses})"
        if (stats is not None and stats.complete)
        else "n/a"
    )

    rows = [
        {
            "Metric": "Intent accuracy",
            "LLM pipeline": _llm_col(lambda m: _fmt_pct(m.intent_accuracy)),
            "Rules baseline": _fmt_pct(rules_m.intent_accuracy),
        },
        {
            "Metric": "Intent macro-F1",
            "LLM pipeline": _llm_col(lambda m: _fmt_pct(m.intent_macro_f1)),
            "Rules baseline": _fmt_pct(rules_m.intent_macro_f1),
        },
        {
            "Metric": "Urgency accuracy",
            "LLM pipeline": _llm_col(lambda m: _fmt_pct(m.urgency_accuracy)),
            "Rules baseline": _fmt_pct(rules_m.urgency_accuracy),
        },
        {
            "Metric": "Urgency macro-F1",
            "LLM pipeline": _llm_col(lambda m: _fmt_pct(m.urgency_macro_f1)),
            "Rules baseline": _fmt_pct(rules_m.urgency_macro_f1),
        },
        {
            "Metric": "Missing-field precision (positive = missing or unclear)",
            "LLM pipeline": _llm_col(lambda m: _fmt_pct(m.detection.precision)),
            "Rules baseline": _fmt_pct(rules_m.detection.precision),
        },
        {
            "Metric": "Missing-field recall (positive = missing or unclear)",
            "LLM pipeline": _llm_col(lambda m: _fmt_pct(m.detection.recall)),
            "Rules baseline": _fmt_pct(rules_m.detection.recall),
        },
        {
            "Metric": "Invalid / malformed LLM response rate",
            "LLM pipeline": inv_str,
            "Rules baseline": "n/a (deterministic)",
        },
        {
            "Metric": "Median latency per inquiry (full pipeline)",
            "LLM pipeline": _llm_col(lambda m: _fmt_ms(m.median_latency_ms)),
            "Rules baseline": _fmt_ms(rules_m.median_latency_ms),
        },
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Intent Confusion Matrices (Rows = Labelled, Columns = Predicted)")
    if llm_m is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_chart_confusion_matrix(llm_m.intent_confusion, "LLM pipeline"), width="stretch")
        with c2:
            st.plotly_chart(_chart_confusion_matrix(rules_m.intent_confusion, "Rules baseline"), width="stretch")
    else:
        st.plotly_chart(_chart_confusion_matrix(rules_m.intent_confusion, "Rules baseline"), width="stretch")

    with st.expander("Per-field missing / unclear detection breakdown"):
        if llm_m is not None:
            st.markdown("**LLM pipeline**")
            st.dataframe(llm_m.per_field, width="stretch", hide_index=True)
        st.markdown("**Rules baseline**")
        st.dataframe(rules_m.per_field, width="stretch", hide_index=True)

    st.subheader("Failure Cases (Heuristic Diagnosis)")
    st.caption(
        "The 'likely why' notes below are **heuristic** starting points derived from surface patterns in the message "
        "and predictions, not a verified root-cause analysis."
    )
    if not report.failures:
        st.info("No failure cases found on the labelled set.")
    for fc in report.failures:
        _render_failure_case(fc, checklist)


def _render_failure_case(fc: FailureCase, checklist: Checklist) -> None:
    """Render one failure case expander with message, labels vs predictions, errors, and heuristic notes."""
    snippet = " ".join(fc.message.split())[:65]
    title = f"{fc.eval_id} · {len(fc.errors)} mismatch(es) — {md(snippet)}"
    with st.expander(title, expanded=False):
        st.markdown("**Customer message**")
        message_box(fc.message)

        comp_rows = [
            {
                "Item": "Intent",
                "Human label": fc.label["intent"],
                "LLM pipeline": fc.llm.intent if fc.llm else "—",
                "Rules baseline": fc.rules.intent,
            },
            {
                "Item": "Urgency",
                "Human label": fc.label["urgency"],
                "LLM pipeline": fc.llm.urgency if fc.llm else "—",
                "Rules baseline": fc.rules.urgency,
            },
        ]
        for f in checklist.fields:
            lbl_s = fc.label[f"status_{f.key}"]
            r_s = fc.rules.statuses[f.key]
            l_s = fc.llm.statuses[f.key] if fc.llm else "—"
            if lbl_s != r_s or (fc.llm and lbl_s != l_s):
                comp_rows.append(
                    {
                        "Item": f.label,
                        "Human label": lbl_s,
                        "LLM pipeline": l_s,
                        "Rules baseline": r_s,
                    }
                )
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

        st.markdown("**What went wrong**")
        for err in fc.errors:
            st.markdown(f"- {md(err)}")

        st.markdown("**Heuristic 'likely why' notes (heuristic only, not verified root causes)**")
        for note in fc.diagnosis:
            st.markdown(f"- {md(note)}")


def render_eval_page(checklist: Checklist) -> None:
    """Render Page 4: Model Evaluation."""
    render_page_header(
        "Model Evaluation",
        "Offline evaluation comparing the deployed LLM pipeline against the rules baseline on the 80-message evaluation set.",
    )
    st.warning(
        "**Small-sample notice:** This evaluation set contains about 80 synthetic messages labelled by hand, so "
        "all metrics are indicative only. Because the messages come from the same generator used while building "
        "the rules baseline, they are likely easier than real customer inquiries."
    )

    report = build_report(checklist)
    for note in report.notes:
        st.info(md(note))

    _render_status_row(report)
    st.divider()

    if not report.label_state.ready:
        st.warning(
            f"**{md(report.label_state.message)}** "
            "No accuracy, F1, precision, or recall figures are shown or estimated until `eval/eval_labeled.csv` "
            "is complete and valid."
        )
        for err in report.label_state.errors:
            st.error(md(err))

        with st.expander("How to label the evaluation set"):
            st.markdown(
                "1. Open `eval/eval_unlabeled.csv` in your spreadsheet editor (or download it below).\n"
                "2. Fill in the empty label columns (`intent`, `urgency`, and one `status_<field>` column for each "
                "of the 12 checklist fields) following `eval/LABELING_GUIDE.md`.\n"
                "3. Save the completed file as `eval/eval_labeled.csv` (UTF-8 CSV) and reload this page. "
                "Nothing in this project ever pre-fills, suggests, or generates labels."
            )
        unl = unlabeled_path()
        if unl.exists():
            st.download_button(
                "Download eval_unlabeled.csv",
                data=unl.read_bytes(),
                file_name="eval_unlabeled.csv",
                mime="text/csv",
                key="btn_download_unlabeled",
            )
        st.divider()

    _render_cached_run_measurements(report)

    if report.label_state.ready and report.rules is not None:
        st.divider()
        _render_labeled_comparison(report, checklist)

    st.divider()
    with st.expander("Evaluation Limitations"):
        st.markdown(
            "- **Synthetic text:** Evaluation messages are generated rather than written by real trucking customers.\n"
            "- **Shared design of baseline and generator:** The rules baseline and synthetic message generator were "
            "developed together, which likely inflates rules performance compared with real-world text.\n"
            "- **Single labeller & small sample:** 80 messages labelled by one person; treat all metrics as indicative.\n"
            "- **Deployed LLM pipeline semantics:** 'LLM pipeline' evaluates the full deployed path—schema validation, "
            "verbatim evidence-quote checks, and automatic fallback to the rules baseline when a message is too short "
            "or an API response is unusable.\n"
            "- **Cached LLM run:** Results come from one cached run in `eval/llm_cache.json` so building the report "
            "is reproducible and makes zero live API calls.\n"
            "- **Two latency definitions:** API-call latency measures only the historical network call for messages "
            "sent to the model; full-pipeline latency includes local validation, quote verification, and rules "
            "fallback or short-message routing."
        )
