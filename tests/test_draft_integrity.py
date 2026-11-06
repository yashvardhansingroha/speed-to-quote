"""Draft integrity: stale drafts, LLM wording vs field IDs, blank corrections, approval-time content rules,
and schema migration."""
import json
import sqlite3
from datetime import datetime

import pytest

from src.analysis import draft_context, load_analysis
from src.config import load_checklist
from src.data_generation.build_database import build_database, ensure_database
from src.database import db, repository as repo
from src.drafting import workflow
from src.drafting.draft import generate_draft, template_draft
from tests.test_pipeline_fallback import StubClient

T0 = datetime(2026, 11, 2, 9, 0)
CHECKLIST = load_checklist()


@pytest.fixture()
def conn(tmp_path):
    path = tmp_path / "p2.db"
    build_database(path, n=25, seed=77)
    c = db.connect(path)
    yield c
    c.close()


def ctx(conn, iid):
    return draft_context(conn, load_analysis(conn, iid, T0), CHECKLIST)


def inquiry_with(conn, key, status):
    return conn.execute("SELECT inquiry_id FROM checklist_status WHERE field_key=? AND status=? LIMIT 1",
                        (key, status)).fetchone()[0]


def make_draft(conn, iid, body=None):
    c = ctx(conn, iid)
    return workflow.create_draft(conn, iid, body or template_draft(c), "template", T0,
                                 checklist_hash=workflow.checklist_fingerprint(c.statuses))


def approves(conn, iid):
    return conn.execute("SELECT COUNT(*) FROM audit_log WHERE inquiry_id=? AND action='approve'", (iid,)).fetchone()[0]


# ---- 1. drafts go stale when the checklist changes ---------------------------------------------------
def test_fresh_draft_is_not_stale_and_can_be_approved(conn):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    did = make_draft(conn, iid)
    assert not workflow.is_stale(workflow.get_current_draft(conn, iid), ctx(conn, iid).statuses)
    workflow.approve_draft(conn, did, workflow.get_current_draft(conn, iid)["body"], T0, ctx(conn, iid))
    assert approves(conn, iid) == 1


def test_missing_to_provided_makes_an_unapproved_draft_stale_and_unapprovable(conn):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    did = make_draft(conn, iid)
    assert CHECKLIST.get("usdot_mc_number").ask in workflow.get_current_draft(conn, iid)["body"]
    repo.save_correction(conn, iid, "usdot_mc_number", "provided", "1234567", T0)
    assert workflow.is_stale(workflow.get_current_draft(conn, iid), ctx(conn, iid).statuses)
    with pytest.raises(workflow.DraftError, match="checklist has changed"):
        workflow.approve_draft(conn, did, workflow.get_current_draft(conn, iid)["body"], T0, ctx(conn, iid))
    assert approves(conn, iid) == 0
    assert conn.execute("SELECT status FROM drafts WHERE id=?", (did,)).fetchone()[0] == "draft"


def test_provided_to_missing_after_approval_marks_the_approved_draft_outdated_but_keeps_it(conn):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    did = make_draft(conn, iid)
    workflow.approve_draft(conn, did, workflow.get_current_draft(conn, iid)["body"], T0, ctx(conn, iid))
    other = next(k for k in CHECKLIST.keys if ctx(conn, iid).statuses[k] == "provided") \
        if "provided" in ctx(conn, iid).statuses.values() else None
    key = other or "fleet_size"
    repo.save_correction(conn, iid, key, "missing", None, T0)
    cur = workflow.get_current_draft(conn, iid)
    assert cur["status"] == "approved"                                        # history is preserved ...
    assert workflow.is_stale(cur, ctx(conn, iid).statuses)                    # ... but flagged outdated
    fresh = make_draft(conn, iid)                                             # a new draft for the current state
    assert workflow.get_current_draft(conn, iid)["id"] == fresh and fresh != did
    workflow.approve_draft(conn, fresh, workflow.get_current_draft(conn, iid)["body"], T0, ctx(conn, iid))
    assert approves(conn, iid) == 2


def test_value_only_corrections_do_not_make_a_draft_stale(conn):
    iid = inquiry_with(conn, "fleet_size", "provided") if conn.execute(
        "SELECT 1 FROM checklist_status WHERE field_key='fleet_size' AND status='provided'").fetchone() else 1
    make_draft(conn, iid)
    status = ctx(conn, iid).statuses["fleet_size"]
    repo.save_correction(conn, iid, "fleet_size", status if status != "provided" else "provided",
                         "77" if status == "provided" else None, T0)
    assert not workflow.is_stale(workflow.get_current_draft(conn, iid), ctx(conn, iid).statuses)


def test_a_draft_with_no_fingerprint_is_treated_as_outdated(conn):
    did = workflow.create_draft(conn, 1, "body", "template", T0)             # e.g. created before this feature existed
    assert workflow.is_stale(workflow.get_current_draft(conn, 1), ctx(conn, 1).statuses)
    with pytest.raises(workflow.DraftError):
        workflow.approve_draft(conn, did, "ok", T0, ctx(conn, 1))


def test_stale_drafts_can_still_be_saved_or_rejected(conn):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    did = make_draft(conn, iid)
    repo.save_correction(conn, iid, "usdot_mc_number", "provided", "1234567", T0)
    workflow.save_draft(conn, did, "edited while outdated", T0)
    workflow.reject_draft(conn, did, "edited while outdated", T0, "outdated")
    assert conn.execute("SELECT status FROM drafts WHERE id=?", (did,)).fetchone()[0] == "rejected"


# ---- 2. LLM wording cannot contradict the field IDs ---------------------------------------------------
def only_usdot_ctx(conn):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    c = ctx(conn, iid)
    c.statuses = {k: ("missing" if k == "usdot_mc_number" else "provided") for k in CHECKLIST.keys}
    return c


def test_an_extra_request_in_the_models_introduction_cannot_enter_the_draft(conn):
    """Fleet size is provided and only USDOT is missing, but the model's intro asks about trucks: that request must not appear."""
    c = only_usdot_ctx(conn)
    answer = {"greeting": "Hi!", "intro": "Please confirm how many trucks you operate.",
              "closing": "Also tell us your fleet size.", "groups": [{"field_keys": ["usdot_mc_number"],
                                                                     "title": "Fleet", "text": "How many trucks do you operate?"}]}
    res = generate_draft(c, StubClient(answer=json.dumps(answer)))
    assert res.generated_by == "llm"
    assert CHECKLIST.get("usdot_mc_number").ask in res.body                  # the one real request is there
    for leaked in ("how many trucks", "fleet size", "Fleet"):
        assert leaked.lower() not in res.body.lower().replace(CHECKLIST.get("usdot_mc_number").ask.lower(), "")
    asks = [f.ask for f in CHECKLIST.fields if f.ask in res.body]
    assert asks == [CHECKLIST.get("usdot_mc_number").ask]                    # exactly one request, and it is the right one


def test_headings_come_from_the_configured_groups(conn):
    c = only_usdot_ctx(conn)
    res = generate_draft(c, StubClient(answer=json.dumps({"groups": [{"field_keys": ["usdot_mc_number"], "title": "Anything"}]})))
    assert f"• {CHECKLIST.get('usdot_mc_number').group}:" in res.body and "Anything" not in res.body


def test_a_mixed_group_gets_the_configured_headings_of_its_fields(conn):
    iid = 1
    c = ctx(conn, iid)
    c.statuses = {k: "provided" for k in CHECKLIST.keys}
    c.statuses.update({"usdot_mc_number": "missing", "fleet_size": "missing"})
    res = generate_draft(c, StubClient(answer=json.dumps({"groups": [{"field_keys": ["usdot_mc_number", "fleet_size"]}]})))
    assert "• Business details / Fleet and drivers:" in res.body


def test_llm_has_no_text_slot_in_its_schema_or_prompt():
    from src.drafting.draft import _DRAFT_SCHEMA, _DRAFT_SYSTEM
    assert set(_DRAFT_SCHEMA["properties"]) == {"groups"}
    assert "Do NOT write any greeting" in _DRAFT_SYSTEM


# ---- 3. a blank 'provided' correction cannot raise completeness ---------------------------------------
@pytest.mark.parametrize("blank", [None, "", "   "])
def test_provided_correction_requires_a_value_and_writes_nothing(conn, blank):
    iid = inquiry_with(conn, "business_legal_name", "missing")
    before = repo.checklist_rows(conn, iid)["business_legal_name"]["status"]
    n_audit = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    with pytest.raises(ValueError, match="value is required"):
        repo.save_correction(conn, iid, "business_legal_name", "provided", blank, T0)
    assert repo.checklist_rows(conn, iid)["business_legal_name"]["status"] == before == "missing"
    assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == n_audit


def test_other_statuses_do_not_need_a_value(conn):
    iid = inquiry_with(conn, "business_legal_name", "missing")
    repo.save_correction(conn, iid, "business_legal_name", "unclear", "", T0)
    repo.save_correction(conn, iid, "business_legal_name", "not_applicable", "", T0)
    repo.save_correction(conn, iid, "business_legal_name", "missing", "", T0)


# ---- 4. approval enforces the content rules (backend) --------------------------------------------------
BAD = ["Your premium is $500 and we guarantee coverage today.", "We will cover you within 24 hours.",
       "You are eligible. Please reply by December 9."]


@pytest.mark.parametrize("text", BAD)
def test_prohibited_text_cannot_be_approved_and_leaves_no_approval_audit(conn, text):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    did = make_draft(conn, iid)
    with pytest.raises(workflow.DraftError, match="cannot be approved"):
        workflow.approve_draft(conn, did, text, T0, ctx(conn, iid))
    row = conn.execute("SELECT * FROM drafts WHERE id=?", (did,)).fetchone()
    assert row["status"] == "draft" and row["body"] == row["original_body"]       # nothing was stored
    assert approves(conn, iid) == 0


def test_prohibited_text_can_still_be_saved_for_later_or_rejected(conn):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    did = make_draft(conn, iid)
    workflow.save_draft(conn, did, BAD[0], T0)
    assert conn.execute("SELECT body FROM drafts WHERE id=?", (did,)).fetchone()[0] == BAD[0]
    workflow.reject_draft(conn, did, BAD[0], T0, "mentions price")
    assert conn.execute("SELECT status FROM drafts WHERE id=?", (did,)).fetchone()[0] == "rejected"


def test_the_signature_placeholder_and_ask_again_heuristic_do_not_block_approval(conn):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    did = make_draft(conn, iid)                                                  # still contains [Your name]
    text = "Thanks for the details about your fleet size. Could you send your USDOT number?\n[Your name]"
    workflow.approve_draft(conn, did, text, T0, ctx(conn, iid))
    assert conn.execute("SELECT status FROM drafts WHERE id=?", (did,)).fetchone()[0] == "approved"


def test_fixed_text_can_be_approved_after_a_refusal(conn):
    iid = inquiry_with(conn, "usdot_mc_number", "missing")
    did = make_draft(conn, iid)
    with pytest.raises(workflow.DraftError):
        workflow.approve_draft(conn, did, BAD[0], T0, ctx(conn, iid))
    workflow.approve_draft(conn, did, "Could you send your USDOT number? Thank you, Dana.", T0, ctx(conn, iid))
    assert approves(conn, iid) == 1


# ---- migration: schema upgrades keep user data ---------------------------------------------------------
def test_migration_from_v2_keeps_drafts_and_audit_rows(tmp_path):
    path = tmp_path / "old.db"
    build_database(path, n=10, seed=3)
    c = db.connect(path)
    did = workflow.create_draft(c, 1, "kept draft", "template", T0, checklist_hash="abc")
    workflow.approve_draft(c, did, "kept draft", T0, ctx(c, 1)) if False else None
    c.execute("ALTER TABLE drafts DROP COLUMN checklist_hash")                   # recreate the v2 layout
    c.execute("PRAGMA user_version = 2")
    c.commit()
    c.close()
    assert not db.schema_is_current(path)
    ensure_database(path)                                                        # migrates, does NOT rebuild
    c = db.connect(path)
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    row = c.execute("SELECT * FROM drafts WHERE id=?", (did,)).fetchone()
    assert row["body"] == "kept draft" and row["checklist_hash"] == ""
    assert c.execute("SELECT COUNT(*) FROM audit_log WHERE action='generate_draft'").fetchone()[0] == 1
    assert workflow.is_stale(row, ctx(c, 1).statuses)                            # unknown fingerprint -> outdated
    c.close()


def test_an_unmigratable_old_database_is_rebuilt(tmp_path):
    path = tmp_path / "ancient.db"
    build_database(path, n=5, seed=3)
    c = sqlite3.connect(path)
    c.execute("PRAGMA user_version = 1")
    c.commit()
    c.close()
    ensure_database(path)
    assert db.schema_is_current(path)
