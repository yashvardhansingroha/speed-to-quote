"""Database integrity and event-sequence validity for the synthetic data."""
from datetime import datetime

from src.database import db

TERMINAL = {"bound", "lost", "closed"}
OUTCOMES = {"bound", "lost"}


def _ts(s):
    return datetime.fromisoformat(s)


def test_all_tables_exist(built_db):
    names = {r[0] for r in built_db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(db.TABLES) <= names


def test_foreign_keys_are_consistent(built_db):
    assert built_db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_every_inquiry_is_synthetic_with_placeholder_contacts(built_db):
    rows = built_db.execute("SELECT contact_email, contact_phone, is_synthetic FROM inquiries").fetchall()
    assert len(rows) == 90
    assert all(r["is_synthetic"] == 1 for r in rows)
    assert all(r["contact_email"].endswith("@example.com") for r in rows)
    assert all(r["contact_phone"].startswith("555-01") for r in rows)


def test_every_inquiry_has_extraction_and_full_checklist(built_db, checklist):
    n = built_db.execute("SELECT COUNT(*) FROM inquiries").fetchone()[0]
    assert built_db.execute("SELECT COUNT(DISTINCT inquiry_id) FROM extractions").fetchone()[0] == n
    per = built_db.execute("SELECT COUNT(*) c FROM checklist_status GROUP BY inquiry_id").fetchall()
    assert all(r["c"] == len(checklist.fields) for r in per)


def test_event_sequences_are_valid(built_db, now):
    inquiries = built_db.execute("SELECT id, received_at, status FROM inquiries").fetchall()
    for inq in inquiries:
        events = built_db.execute(
            "SELECT event_type, occurred_at FROM events WHERE inquiry_id=? ORDER BY occurred_at, id", (inq["id"],)
        ).fetchall()
        kinds = [e["event_type"] for e in events]
        times = [_ts(e["occurred_at"]) for e in events]

        # starts with creation at the received time; everything is in order and not in the future
        assert kinds[0] == "inquiry_created" and times[0] == _ts(inq["received_at"])
        assert times == sorted(times)
        assert all(t <= now for t in times)
        assert kinds.count("inquiry_created") == 1 and kinds.count("first_response") <= 1

        # no response before creation
        if "first_response" in kinds:
            assert times[kinds.index("first_response")] > times[0]
            assert kinds.index("first_response") < min(
                [i for i, k in enumerate(kinds) if k in ("followup_sent", "customer_reply", "quote_sent")] or [99])

        # no outcome without an earlier quote stage; at most one terminal event, and it is last
        terminals = [i for i, k in enumerate(kinds) if k in TERMINAL]
        assert len(terminals) <= 1
        if terminals:
            assert terminals[0] == len(kinds) - 1
        for i, k in enumerate(kinds):
            if k in OUTCOMES:
                assert "quote_sent" in kinds[:i], f"inquiry {inq['id']}: {k} without prior quote_sent"
        if "quote_sent" in kinds:
            assert "closed" not in kinds                # 'closed' is only for non-quote inquiries

        # the denormalised status column matches the events
        expected = kinds[terminals[0]] if terminals else "open"
        assert inq["status"] == expected


def test_followup_tasks_are_consistent(built_db):
    rows = built_db.execute(
        """SELECT t.status tstatus, t.created_at, t.due_at, t.completed_at, i.status istatus
           FROM followup_tasks t JOIN inquiries i ON i.id = t.inquiry_id""").fetchall()
    assert rows, "expected some follow-up tasks"
    for r in rows:
        assert _ts(r["due_at"]) > _ts(r["created_at"])
        if r["tstatus"] == "open":
            assert r["istatus"] == "open", "open follow-ups may only belong to open inquiries"
            assert r["completed_at"] is None
        if r["tstatus"] in ("done", "cancelled"):
            assert r["completed_at"] is not None


def test_status_check_constraint_rejects_bad_checklist_status(built_db):
    import sqlite3
    import pytest
    with pytest.raises(sqlite3.IntegrityError):
        built_db.execute(
            "INSERT INTO checklist_status (inquiry_id, field_key, status, updated_at) VALUES (1,'x','bogus','2026-01-01T00:00:00')")
    built_db.rollback()
