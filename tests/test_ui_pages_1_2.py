"""UI tests for Page 1 (Inquiry Triage) and Page 2 (Submission Review)."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.data_generation.build_database import build_database
from src.database import db
from src.database import repository as repo
from src.drafting import workflow
from src.scoring.completeness import READINESS_NOTE

APP_PATH = str((ROOT / "streamlit_app.py").resolve())

EXAMPLE_PASTE = (
    "Hi, we run a trucking business with 12 trucks operating in Texas and Oklahoma. "
    "Our insurance expires next month. We're looking for a new quote and want to talk to someone this week."
)


@pytest.fixture()
def ui_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "pages_1_2.db"
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


def _submit_correction(at: AppTest, inquiry_id: int, field_key: str) -> None:
    target_form = f"fix_{inquiry_id}_{field_key}"
    for btn in at.button:
        key = getattr(btn, "key", "") or ""
        form_id = getattr(btn, "form_id", "") or ""
        if target_form in key or form_id == target_form:
            btn.click()
            return
    raise AssertionError(f"Submit button for form {target_form} not found")


def test_page1_stored_inquiry_and_paste_flow_and_navigation(ui_db: Path) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Stored inquiry renders either a score or human-review notice plus checklist disclaimer
    text = _page_text(at)
    assert ("Priority score:" in text) or ("Routed to human review: not scored" in text)
    assert "operational urgency" in text.lower()

    # Switch to Paste mode and verify max_chars = 10000 and privacy warning
    at.radio(key="triage_source").set_value("Paste a message").run()
    assert not at.exception
    paste_box = at.text_area(key="paste_message_text")
    assert paste_box.max_chars == 10000
    assert "Do not paste real customer or personal information" in _page_text(at)

    # Paste the spec example -> score 65, factor reasons, operational-urgency note
    conn = db.connect(ui_db)
    count_before = conn.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0]

    paste_box.set_value(EXAMPLE_PASTE)
    at.button(key="btn_analyze_paste").click().run()
    assert not at.exception
    pasted_text = _page_text(at)
    assert "65 / 100" in pasted_text
    assert "not a prediction of conversion" in pasted_text
    assert len(at.dataframe) >= 2

    count_after_first = conn.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0]
    assert count_after_first == count_before + 1

    # Clicking Analyze a second time on the exact same text must re-analyse the same inquiry, not duplicate
    at.button(key="btn_analyze_paste").click().run()
    assert not at.exception
    count_after_second = conn.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0]
    assert count_after_second == count_after_first
    pasted_id = int(at.session_state["current_inquiry_id"])
    conn.close()

    # "Open in Submission Review" navigates to Page 2 and keeps the inquiry selected
    at.button(key="btn_open_in_review").click().run()
    assert not at.exception
    assert at.session_state["nav"] == "Submission Review"
    assert int(at.session_state["current_inquiry_id"]) == pasted_id


def test_page1_empty_and_short_pastes_routed_to_review(ui_db: Path) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.radio(key="triage_source").set_value("Paste a message").run()

    for short_msg in ("", "hi there"):
        at.text_area(key="paste_message_text").set_value(short_msg)
        at.button(key="btn_analyze_paste").click().run()
        assert not at.exception
        text = _page_text(at)
        assert "Routed to human review: not scored" in text
        assert "Priority score:" not in text


def test_page2_completeness_and_corrections(ui_db: Path) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav").set_value("Submission Review").run()
    assert not at.exception

    inq_id = int(at.session_state["current_inquiry_id"])
    text = _page_text(at)
    assert "Submission completeness:" in text
    assert READINESS_NOTE in text

    # Refuses blank "provided" correction with error at top and nothing written to DB
    conn = db.connect(ui_db)
    audit_before = len(workflow.audit_history(conn, inq_id))
    at.selectbox(key=f"fix_status_{inq_id}_usdot_mc_number").set_value("provided")
    at.text_input(key=f"fix_value_{inq_id}_usdot_mc_number").set_value("   ")
    _submit_correction(at, inq_id, "usdot_mc_number")
    at.run()
    assert not at.exception
    assert any("A value is required when marking a field as provided" in str(e.value) for e in at.error)
    assert len(workflow.audit_history(conn, inq_id)) == audit_before

    # Valid manual correction persists with source='manual', writes audit row, and updates completeness
    at.selectbox(key=f"fix_status_{inq_id}_usdot_mc_number").set_value("provided")
    at.text_input(key=f"fix_value_{inq_id}_usdot_mc_number").set_value("USDOT 998877")
    at.text_input(key=f"fix_reason_{inq_id}_usdot_mc_number").set_value("Confirmed on call")
    _submit_correction(at, inq_id, "usdot_mc_number")
    at.run()
    assert not at.exception

    rows = repo.checklist_rows(conn, inq_id)
    assert rows["usdot_mc_number"]["status"] == "provided"
    assert rows["usdot_mc_number"]["value"] == "USDOT 998877"
    assert rows["usdot_mc_number"]["source"] == "manual"
    history = workflow.audit_history(conn, inq_id)
    assert len(history) == audit_before + 1
    assert history[-1]["action"] == "correct_field"
    assert "Confirmed on call" in (history[-1]["reason"] or "")
    conn.close()


def test_page2_draft_lifecycle_safety_and_staleness(ui_db: Path) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav").set_value("Submission Review").run()
    inq_id = int(at.session_state["current_inquiry_id"])

    # Generate initial follow-up draft
    at.button(key=f"btn_gen_draft_{inq_id}").click().run()
    assert not at.exception

    conn = db.connect(ui_db)
    draft = workflow.get_current_draft(conn, inq_id)
    assert draft is not None
    draft_id = int(draft["id"])

    # Empty draft disables Approve
    at.text_area(key=f"draft_text_{draft_id}").set_value("   ").run()
    assert at.button(key=f"btn_approve_{draft_id}").disabled is True
    assert any("empty" in str(e.value).lower() for e in at.error)

    # Banned text (pricing / guarantee / time commitment) disables Approve and shows why
    at.text_area(key=f"draft_text_{draft_id}").set_value(
        "Hello, we guarantee the lowest price of $900 within 2 hours."
    ).run()
    assert at.button(key=f"btn_approve_{draft_id}").disabled is True
    assert any("cannot be approved" in str(e.value).lower() for e in at.error)

    # Restore clean text, replace placeholder signature, and Save draft
    clean_text = draft["body"].replace("[Your name]", "Alex Broker")
    at.text_area(key=f"draft_text_{draft_id}").set_value(clean_text)
    at.button(key=f"btn_save_{draft_id}").click().run()
    assert not at.exception
    assert workflow.get_current_draft(conn, inq_id)["body"] == clean_text

    # Reject draft with a reason -> persists status='rejected' and reason in audit log
    at.text_input(key=f"draft_reason_{draft_id}").set_value("Needs supervisor check")
    at.button(key=f"btn_reject_{draft_id}").click().run()
    assert not at.exception
    assert workflow.get_current_draft(conn, inq_id)["status"] == "rejected"
    assert "Needs supervisor check" in _page_text(at)

    # Generate a fresh draft and Approve it -> approved text is NOT shown in a disabled text_area
    at.button(key=f"btn_new_draft_{draft_id}").click().run()
    draft2 = workflow.get_current_draft(conn, inq_id)
    draft2_id = int(draft2["id"])
    at.button(key=f"btn_approve_{draft2_id}").click().run()
    assert not at.exception
    assert workflow.get_current_draft(conn, inq_id)["status"] == "approved"
    assert len(at.text_area) == 0

    # Generate a third draft, then change a checklist status -> draft becomes Outdated and cannot be approved
    at.button(key=f"btn_new_draft_{draft2_id}").click().run()
    draft3_id = int(workflow.get_current_draft(conn, inq_id)["id"])
    target_field = "loss_runs"
    cur_status = repo.checklist_rows(conn, inq_id)[target_field]["status"]
    flipped = "not_applicable" if cur_status != "not_applicable" else "missing"
    at.selectbox(key=f"fix_status_{inq_id}_{target_field}").set_value(flipped)
    _submit_correction(at, inq_id, target_field)
    at.run()
    assert not at.exception
    assert "Outdated" in _page_text(at)
    assert at.button(key=f"btn_approve_{draft3_id}").disabled is True
    conn.close()


def test_empty_database_and_hostile_markdown_escaping_on_pages_1_and_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_db = tmp_path / "empty.db"
    conn = db.connect(empty_db)
    db.init_db(conn)
    conn.close()
    monkeypatch.setenv("STQ_DB_PATH", str(empty_db))
    monkeypatch.setenv("STQ_FORCE_OFFLINE", "1")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    at.sidebar.radio(key="nav").set_value("Submission Review").run()
    assert not at.exception

    # Paste a message containing hostile markdown/HTML links and verify it is escaped
    at.sidebar.radio(key="nav").set_value("Inquiry Triage").run()
    at.radio(key="triage_source").set_value("Paste a message").run()
    hostile = "We run 10 trucks in Texas. [click](http://evil.example) <script>alert(1)</script> Need a new quote ASAP."
    at.text_area(key="paste_message_text").set_value(hostile)
    at.button(key="btn_analyze_paste").click().run()
    assert not at.exception
    raw_out = _page_text(at)
    assert "<script>" not in raw_out
    assert "&lt;script&gt;" in raw_out


def test_page1_summary_cards_hold_html_escaped_text_without_markdown_backslashes(ui_db: Path) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    cards = [str(m.value) for m in at.markdown if str(m.value).startswith('<div class="stq-kpi-card">')]
    assert len(cards) == 3
    for card in cards:
        # Markdown escapes such as "\_" are shown literally inside an HTML block, so none may appear.
        assert "\\" not in card
