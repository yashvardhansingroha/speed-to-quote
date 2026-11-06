"""UI tests for Page 3 (Operations Overview)."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.analytics import inbox, insights, kpis
from src.config import ROOT, load_settings, reference_now
from src.data_generation.build_database import build_database
from src.database import db

APP_PATH = str((ROOT / "streamlit_app.py").resolve())


@pytest.fixture()
def ui_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "page_3.db"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    build_database(db_path, n=30, seed=5)

    monkeypatch.setenv("STQ_DB_PATH", str(db_path))
    monkeypatch.setenv("STQ_EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("STQ_FORCE_OFFLINE", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    return db_path


def _page_text(at: AppTest) -> str:
    parts: list[str] = []
    for coll in (at.markdown, at.info, at.warning, at.error, at.caption, at.title, at.subheader):
        for el in coll:
            parts.append(str(getattr(el, "value", "")))
    for m in at.metric:
        parts.append(f"{m.label}: {m.value}")
    return "\n".join(parts)


def test_page3_kpis_match_computed_values_and_date_filter_and_incomplete_range(ui_db: Path) -> None:
    conn = db.connect(ui_db)
    all_frames = kpis.load_frames(conn)
    lo, hi = kpis.date_bounds(all_frames)
    now = reference_now()
    mature_days = int(load_settings()["mature_cohort_days"])
    expected_full = kpis.compute_kpis(all_frames, now, mature_days, lo, hi)
    conn.close()

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav").set_value("Operations Overview").run()
    assert not at.exception

    text = _page_text(at)
    assert "SYNTHETIC / DEMONSTRATION DATA" in text
    assert f">{expected_full.total}<" in text
    assert f"{expected_full.median_hours:.1f} h" in text
    assert f"P90: {expected_full.p90_hours:.1f} h" in text
    assert f"{expected_full.awaiting} open and awaiting a first response" in text
    assert f"{expected_full.closed_without_response} closed without one" in text
    assert expected_full.overdue.pct_text in text
    assert f"{expected_full.overdue.fraction_text} follow-up tasks are open and past due" in text
    assert f"{expected_full.completeness_avg * 100:.1f}%" in text
    assert expected_full.mature.bound.pct_text in text
    assert f"{expected_full.mature.bound.fraction_text} mature quote requests bound" in text
    assert f"{expected_full.mature.quoted.fraction_text} reached a quote" in text

    # Ensure raw HTML KPI cards contain no literal Markdown backslash escapes
    for m in at.markdown:
        val = str(m.value)
        if "stq-kpi-card" in val:
            assert "\\" not in val

    # Narrowing the date range updates the KPI cards to match the filtered computation
    mid = lo + timedelta(days=(hi - lo).days // 2)
    filtered_frames = kpis.filter_period(all_frames, mid, hi)
    expected_mid = kpis.compute_kpis(filtered_frames, now, mature_days, mid, hi)

    at.date_input(key="ops_date_range").set_value((mid, hi)).run()
    assert not at.exception
    mid_text = _page_text(at)
    assert f">{expected_mid.total}<" in mid_text
    assert f"{expected_mid.overdue.fraction_text} follow-up tasks are open and past due" in mid_text

    # Incomplete date range (single date selected) shows an info prompt and does not crash
    at.date_input(key="ops_date_range").set_value((lo,)).run()
    assert not at.exception
    assert any("Pick an end date" in str(i.value) for i in at.info)


def test_page3_insights_inbox_selection_and_overdue_table(ui_db: Path) -> None:
    conn = db.connect(ui_db)
    all_frames = kpis.load_frames(conn)
    lo, hi = kpis.date_bounds(all_frames)
    now = reference_now()
    mature_days = int(load_settings()["mature_cohort_days"])
    expected_kpis = kpis.compute_kpis(all_frames, now, mature_days, lo, hi)
    expected_insights = insights.build_insights(all_frames, expected_kpis, now)
    expected_inbox = inbox.build_inbox(conn, now)
    conn.close()

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav").set_value("Operations Overview").run()
    assert not at.exception

    text = _page_text(at)

    # All four insights and their four required parts appear
    assert len(expected_insights) == 4
    for ins in expected_insights:
        assert ins.observation in text
        assert ins.action in text
        assert ins.metric in text
        for hyp in ins.hypotheses:
            assert hyp in text
    assert "Possible explanations (hypotheses, not findings)" in text

    # Priority inbox dataframe is sorted by score descending
    ranked_df = at.dataframe(key="p3_inbox_df").value
    scores = list(ranked_df["Score"])
    assert scores == sorted(scores, reverse=True)
    assert list(ranked_df["Inquiry"]) == list(expected_inbox.ranked["Inquiry"])

    # Overdue follow-ups table matches the computed overdue_tasks
    overdue_dfs = [
        d.value for d in at.dataframe if list(d.value.columns) == ["Inquiry", "Task", "Was due", "Days overdue"]
    ]
    assert len(overdue_dfs) == 1
    assert len(overdue_dfs[0]) == expected_kpis.overdue.numerator
    assert list(overdue_dfs[0]["Inquiry"]) == [int(x) for x in expected_kpis.overdue_tasks["inquiry_id"]]

    # Selecting an inquiry via the selectbox updates the detail breakdown and opens it in Submission Review
    target_id = int(expected_inbox.ranked.iloc[-1]["Inquiry"])
    target_analysis = expected_inbox.analyses[target_id]
    at.selectbox(key="p3_selected").set_value(target_id).run()
    assert not at.exception

    updated_text = _page_text(at)
    assert f"Inquiry #{target_id} Breakdown" in updated_text
    assert f"{target_analysis.triage.priority.score} / 100" in updated_text
    factor_tables = [d.value for d in at.dataframe if list(d.value.columns) == ["Factor", "Points", "Why"]]
    assert len(factor_tables) == 1
    assert list(factor_tables[0]["Why"]) == [f.reason for f in target_analysis.triage.priority.factors]

    at.button(key="btn_p3_open_review").click().run()
    assert not at.exception
    assert at.session_state["nav"] == "Submission Review"
    assert int(at.session_state["current_inquiry_id"]) == target_id


def test_page3_empty_database_does_not_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty_db = tmp_path / "empty_ops.db"
    conn = db.connect(empty_db)
    db.init_db(conn)
    conn.close()
    monkeypatch.setenv("STQ_DB_PATH", str(empty_db))
    monkeypatch.setenv("STQ_FORCE_OFFLINE", "1")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav").set_value("Operations Overview").run()
    assert not at.exception
    assert any("No inquiries" in str(i.value) for i in at.info)
