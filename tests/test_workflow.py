"""Draft approve / reject / save persist to the audit log; manual corrections persist and are respected."""
from datetime import datetime, timedelta

import pytest

from src.analysis import analyze_inquiry, draft_context, load_analysis
from src.config import load_checklist
from src.data_generation.build_database import build_database
from src.database import db, repository as repo
from src.drafting import workflow
from src.drafting.draft import generate_draft, template_draft
from src.extraction.llm_client import LLMClient, LLMConfig

T0 = datetime(2026, 11, 2, 9, 0)
OFFLINE = LLMClient(LLMConfig(api_key=None))
CHECKLIST = load_checklist()


def ctx_for(conn, inquiry_id=1):
    """The current draft context (checklist state + message) for an inquiry."""
    return draft_context(conn, load_analysis(conn, inquiry_id, T0), CHECKLIST)


@pytest.fixture()
def conn(tmp_path):
    path = tmp_path / "w.db"
    build_database(path, n=25, seed=77)
    c = db.connect(path)
    yield c
    c.close()


def a_draft(conn, inquiry_id=1, body="Hello,\n\n• Operations: the states you operate in\n\nThank you,\n[Your name]"):
    h = workflow.checklist_fingerprint(ctx_for(conn, inquiry_id).statuses)
    return workflow.create_draft(conn, inquiry_id, body, "template", T0, "note", checklist_hash=h)


def audit(conn, inquiry_id=1):
    return [dict(r) for r in workflow.audit_history(conn, inquiry_id)]


# ---- drafts --------------------------------------------------------------------------------------------
def test_create_draft_stores_original_and_logs_generation(conn):
    did = a_draft(conn)
    row = conn.execute("SELECT * FROM drafts WHERE id=?", (did,)).fetchone()
    assert row["status"] == "draft" and row["body"] == row["original_body"] and row["generated_by"] == "template"
    log = audit(conn)
    assert [r["action"] for r in log] == ["generate_draft"]
    assert log[0]["original_draft"] == row["original_body"] and log[0]["ts"] == "2026-11-02T09:00:00"


def test_save_draft_persists_edit_keeps_original_and_logs(conn):
    did = a_draft(conn)
    workflow.save_draft(conn, did, "EDITED TEXT", T0 + timedelta(minutes=5))
    row = conn.execute("SELECT * FROM drafts WHERE id=?", (did,)).fetchone()
    assert row["status"] == "draft" and row["body"] == "EDITED TEXT" and row["original_body"].startswith("Hello")
    last = audit(conn)[-1]
    assert (last["action"], last["edited_draft"], last["inquiry_id"]) == ("save_draft", "EDITED TEXT", 1)
    assert last["original_draft"] == row["original_body"] and last["ts"] == "2026-11-02T09:05:00"


def test_approve_records_the_human_edited_text(conn):
    did = a_draft(conn)
    workflow.approve_draft(conn, did, "Final wording", T0, ctx_for(conn, 1))
    row = conn.execute("SELECT * FROM drafts WHERE id=?", (did,)).fetchone()
    assert row["status"] == "approved" and row["body"] == "Final wording"
    last = audit(conn)[-1]
    assert last["action"] == "approve" and last["edited_draft"] == "Final wording" and last["reason"] is None


def test_reject_records_the_reason(conn):
    did = a_draft(conn)
    workflow.reject_draft(conn, did, "text at rejection", T0, "  tone is too pushy ")
    assert conn.execute("SELECT status FROM drafts WHERE id=?", (did,)).fetchone()[0] == "rejected"
    last = audit(conn)[-1]
    assert last["action"] == "reject" and last["reason"] == "tone is too pushy"


def test_reject_without_a_reason_is_allowed(conn):
    did = a_draft(conn)
    workflow.reject_draft(conn, did, "text", T0)
    assert audit(conn)[-1]["reason"] is None


def test_every_action_is_logged_in_order_with_all_columns(conn):
    did = a_draft(conn)
    workflow.save_draft(conn, did, "v2", T0 + timedelta(minutes=1))
    workflow.save_draft(conn, did, "v3", T0 + timedelta(minutes=2))
    workflow.approve_draft(conn, did, "v3", T0 + timedelta(minutes=3), ctx_for(conn, 1))
    log = audit(conn)
    assert [r["action"] for r in log] == ["generate_draft", "save_draft", "save_draft", "approve"]
    assert [r["ts"] for r in log] == sorted(r["ts"] for r in log)
    assert all(set(r) >= {"ts", "inquiry_id", "action", "original_draft", "edited_draft", "reason"} for r in log)


def test_finalised_drafts_cannot_be_changed(conn):
    did = a_draft(conn)
    workflow.approve_draft(conn, did, "ok", T0, ctx_for(conn, 1))
    for action in (lambda: workflow.save_draft(conn, did, "x", T0), lambda: workflow.approve_draft(conn, did, "x", T0, ctx_for(conn, 1)),
                   lambda: workflow.reject_draft(conn, did, "x", T0, "r")):
        with pytest.raises(workflow.DraftError, match="already approved"):
            action()
    assert len(audit(conn)) == 2                         # generate + approve only; failed attempts are not logged


def test_empty_text_is_refused_and_nothing_changes(conn):
    did = a_draft(conn)
    for blank in ("", "   \n"):
        with pytest.raises(workflow.DraftError, match="empty"):
            workflow.approve_draft(conn, did, blank, T0, ctx_for(conn, 1))
    assert conn.execute("SELECT status FROM drafts WHERE id=?", (did,)).fetchone()[0] == "draft"
    assert len(audit(conn)) == 1


def test_unknown_draft_is_refused(conn):
    with pytest.raises(workflow.DraftError):
        workflow.approve_draft(conn, 9999, "x", T0, ctx_for(conn, 1))


def test_a_new_draft_becomes_current_and_history_is_kept(conn):
    first = a_draft(conn)
    workflow.reject_draft(conn, first, "x", T0, "no")
    second = a_draft(conn, body="second version")
    assert workflow.get_current_draft(conn, 1)["id"] == second
    assert conn.execute("SELECT COUNT(*) FROM drafts WHERE inquiry_id=1").fetchone()[0] == 2


def test_audit_is_per_inquiry(conn):
    a_draft(conn, 1)
    a_draft(conn, 2)
    assert len(audit(conn, 1)) == 1 and len(audit(conn, 2)) == 1 and audit(conn, 3) == []


def test_no_draft_means_none(conn):
    assert workflow.get_current_draft(conn, 1) is None


# ---- manual corrections -----------------------------------------------------------------------------------
def first_field(conn, status):
    return conn.execute("SELECT inquiry_id, field_key FROM checklist_status WHERE status=? LIMIT 1", (status,)).fetchone()


def test_correction_persists_and_is_marked_manual(conn):
    r = first_field(conn, "missing")
    repo.save_correction(conn, r["inquiry_id"], r["field_key"], "provided", " 14 trucks ", T0, "customer phoned")
    row = repo.checklist_rows(conn, r["inquiry_id"])[r["field_key"]]
    assert (row["status"], row["value"], row["source"]) == ("provided", "14 trucks", "manual")


def test_correction_is_audited_with_before_and_after(conn):
    r = first_field(conn, "missing")
    repo.save_correction(conn, r["inquiry_id"], r["field_key"], "provided", "x", T0, "why")
    last = audit(conn, r["inquiry_id"])[-1]
    assert last["action"] == "correct_field" and last["original_draft"].startswith("missing")
    assert last["edited_draft"] == "provided | x" and last["reason"] == f"{r['field_key']}: why"


def test_missing_and_not_applicable_corrections_drop_the_value(conn):
    r = first_field(conn, "provided")
    repo.save_correction(conn, r["inquiry_id"], r["field_key"], "not_applicable", "stale", T0)
    assert repo.checklist_rows(conn, r["inquiry_id"])[r["field_key"]]["value"] is None


def test_invalid_correction_is_refused(conn):
    r = first_field(conn, "missing")
    with pytest.raises(ValueError):
        repo.save_correction(conn, r["inquiry_id"], r["field_key"], "bogus", "x", T0)
    with pytest.raises(KeyError):
        repo.save_correction(conn, r["inquiry_id"], "no_such_field", "provided", "x", T0)


def test_reanalysis_does_not_overwrite_a_human_correction(conn, checklist):
    r = first_field(conn, "missing")
    repo.save_correction(conn, r["inquiry_id"], r["field_key"], "provided", "human value", T0)
    analyze_inquiry(conn, r["inquiry_id"], checklist, OFFLINE, T0)
    row = repo.checklist_rows(conn, r["inquiry_id"])[r["field_key"]]
    assert row["source"] == "manual" and row["value"] == "human value"
    other = [v for k, v in repo.checklist_rows(conn, r["inquiry_id"]).items() if k != r["field_key"]]
    assert all(v["source"] == "rules" for v in other)          # untouched fields are refreshed by the machine


def test_corrected_fields_change_completeness_and_the_draft(conn, checklist):
    from src.scoring.completeness import compute_completeness
    r = first_field(conn, "missing")
    a = load_analysis(conn, r["inquiry_id"], T0)
    before_ctx = draft_context(conn, a, checklist)
    field = checklist.get(r["field_key"])
    assert field.key in [f.key for f in before_ctx.requested()] and field.ask in template_draft(before_ctx)
    before = compute_completeness(before_ctx.statuses)

    repo.save_correction(conn, r["inquiry_id"], r["field_key"], "provided", "from phone call", T0)
    after_ctx = draft_context(conn, load_analysis(conn, r["inquiry_id"], T0), checklist)
    after = compute_completeness(after_ctx.statuses)
    assert after.provided == before.provided + 1
    assert field.ask not in template_draft(after_ctx)           # a human-provided field is not asked again


def test_offline_pipeline_produces_a_savable_template_draft(conn, checklist):
    a = analyze_inquiry(conn, 1, checklist, OFFLINE, T0)
    result = generate_draft(draft_context(conn, a, checklist), OFFLINE)
    ctx = draft_context(conn, a, CHECKLIST)
    did = workflow.create_draft(conn, 1, result.body, result.generated_by, T0, result.note,
                                checklist_hash=workflow.checklist_fingerprint(ctx.statuses))
    workflow.approve_draft(conn, did, result.body.replace("[Your name]", "Dana Reyes"), T0, ctx)
    assert conn.execute("SELECT status FROM drafts WHERE id=?", (did,)).fetchone()[0] == "approved"


def test_pasted_inquiry_round_trip(conn, checklist):
    iid = repo.create_pasted_inquiry(conn, "We run 5 trucks in Ohio and need a quote.", T0)
    row = repo.get_inquiry(conn, iid)
    assert row["channel"] == "pasted" and row["is_synthetic"] == 0
    a = analyze_inquiry(conn, iid, checklist, OFFLINE, T0)
    assert a.method == "rules" and a.extraction.fields["fleet_size"].value == "5"
    assert conn.execute("SELECT COUNT(*) FROM events WHERE inquiry_id=?", (iid,)).fetchone()[0] == 1
    assert draft_context(conn, a, checklist).customer_first_name is None      # no invented name for pasted text
