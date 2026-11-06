"""Small, explicit read/write helpers on top of sqlite3 (shared by the generator and the app)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from src.extraction.pipeline import ExtractionOutcome
from src.extraction.schema import InquiryExtraction

ISO = "%Y-%m-%dT%H:%M:%S"


def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime(ISO) if dt else None


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    return datetime.strptime(value, ISO) if value else None


def save_extraction(conn: sqlite3.Connection, inquiry_id: int, outcome: ExtractionOutcome,
                    needs_review: bool, review_reason: Optional[str],
                    deadline_date: Optional[str], created_at: datetime,
                    deadline_latest: Optional[str] = None, deadline_approximate: bool = False) -> int:
    """Store one extraction run (overwrites nothing: history is kept, latest row wins)."""
    ex = outcome.extraction
    cur = conn.execute(
        """INSERT INTO extractions (inquiry_id, method, model, intent, urgency, intent_evidence,
               urgency_evidence, business_description, deadline_text, deadline_date, deadline_latest,
               deadline_approximate, needs_review, review_reason, fallback_reason, latency_ms,
               raw_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (inquiry_id, outcome.method, outcome.model, ex.intent, ex.urgency, ex.intent_evidence,
         ex.urgency_evidence, ex.business_description.value, ex.deadline.value, deadline_date,
         deadline_latest, int(deadline_approximate), int(needs_review), review_reason, outcome.fallback_reason, outcome.latency_ms,
         ex.model_dump_json(), iso(created_at)))
    return int(cur.lastrowid)


def save_checklist(conn: sqlite3.Connection, inquiry_id: int, outcome: ExtractionOutcome,
                   updated_at: datetime, keep_manual: bool = True) -> None:
    """Write the machine's checklist result (source = 'llm' or 'rules').

    A human correction is never silently overwritten: with keep_manual=True (the default) any field
    whose current source is 'manual' is left alone when an inquiry is re-analysed.
    """
    manual = set()
    if keep_manual:
        manual = {r["field_key"] for r in conn.execute(
            "SELECT field_key FROM checklist_status WHERE inquiry_id = ? AND source = 'manual'", (inquiry_id,))}
    for key, f in outcome.extraction.fields.items():
        if key in manual:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO checklist_status
                   (inquiry_id, field_key, status, value, evidence, evidence_verified, source, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (inquiry_id, key, f.status, f.value, f.evidence,
             None if f.evidence_verified is None else int(f.evidence_verified),
             outcome.method, iso(updated_at)))


# --------------------------------------------------------------------------- reading
def list_inquiries(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, customer_name, channel, message, received_at, status, is_synthetic "
        "FROM inquiries ORDER BY received_at DESC, id DESC LIMIT ?", (limit,)).fetchall()


def get_inquiry(conn: sqlite3.Connection, inquiry_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM inquiries WHERE id = ?", (inquiry_id,)).fetchone()


def latest_extraction(conn: sqlite3.Connection, inquiry_id: int) -> Optional[tuple[sqlite3.Row, InquiryExtraction]]:
    """The newest stored extraction row and its parsed object (None if never analysed)."""
    row = conn.execute("SELECT * FROM extractions WHERE inquiry_id = ? ORDER BY id DESC LIMIT 1",
                       (inquiry_id,)).fetchone()
    if row is None:
        return None
    return row, InquiryExtraction.model_validate_json(row["raw_json"])


def checklist_rows(conn: sqlite3.Connection, inquiry_id: int) -> dict[str, sqlite3.Row]:
    """Current checklist state per field key (machine results plus any manual corrections)."""
    return {r["field_key"]: r for r in conn.execute(
        "SELECT * FROM checklist_status WHERE inquiry_id = ?", (inquiry_id,))}


def first_response_at(conn: sqlite3.Connection, inquiry_id: int) -> Optional[datetime]:
    r = conn.execute("SELECT occurred_at FROM events WHERE inquiry_id = ? AND event_type = 'first_response' "
                     "ORDER BY occurred_at LIMIT 1", (inquiry_id,)).fetchone()
    return parse_ts(r["occurred_at"]) if r else None


def open_task_due_times(conn: sqlite3.Connection, inquiry_id: int) -> list[datetime]:
    return [parse_ts(r["due_at"]) for r in conn.execute(
        "SELECT due_at FROM followup_tasks WHERE inquiry_id = ? AND status = 'open'", (inquiry_id,))]


# --------------------------------------------------------------------------- writing
def create_pasted_inquiry(conn: sqlite3.Connection, message: str, received_at: datetime,
                          customer_name: str = "Pasted inquiry") -> int:
    """Store a message the user pasted into the app (demo text, not part of the generated set)."""
    cur = conn.execute(
        """INSERT INTO inquiries (customer_name, contact_email, contact_phone, channel, message,
               received_at, status, is_synthetic) VALUES (?,?,?,?,?,?,?,0)""",
        (customer_name, "pasted@example.com", "555-0100", "pasted", message, iso(received_at), "open"))
    conn.execute("INSERT INTO events (inquiry_id, event_type, occurred_at, detail) VALUES (?,?,?,?)",
                 (cur.lastrowid, "inquiry_created", iso(received_at), "pasted in app"))
    conn.commit()
    return int(cur.lastrowid)


VALID_STATUSES = ("provided", "missing", "unclear", "not_applicable")


def save_correction(conn: sqlite3.Connection, inquiry_id: int, field_key: str, status: str,
                    value: Optional[str], ts: datetime, reason: Optional[str] = None) -> None:
    """Record a human correction of one checklist field and log it in the audit trail.

    The supporting quote is kept as it was (it is what the machine saw); the status/value become the
    human's, with source = 'manual'. Audit: original_draft = "<old status> | <old value>",
    edited_draft = "<new status> | <new value>", reason = "<field_key>: <optional note>".
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}")
    old = conn.execute("SELECT * FROM checklist_status WHERE inquiry_id = ? AND field_key = ?",
                       (inquiry_id, field_key)).fetchone()
    if old is None:
        raise KeyError(f"no checklist row for inquiry {inquiry_id}, field {field_key}")
    value = (value or "").strip() or None
    if status == "provided" and not value:
        # Validate BEFORE any write, so a rejected correction changes neither the checklist nor the audit log.
        raise ValueError("A value is required when marking a field as provided.")
    if status in ("missing", "not_applicable"):
        value = None
    conn.execute(
        "UPDATE checklist_status SET status = ?, value = ?, source = 'manual', updated_at = ? "
        "WHERE inquiry_id = ? AND field_key = ?", (status, value, iso(ts), inquiry_id, field_key))
    conn.execute(
        "INSERT INTO audit_log (ts, inquiry_id, action, original_draft, edited_draft, reason) VALUES (?,?,?,?,?,?)",
        (iso(ts), inquiry_id, "correct_field", f"{old['status']} | {old['value'] or ''}",
         f"{status} | {value or ''}", f"{field_key}: {reason}" if reason else field_key))
    conn.commit()
