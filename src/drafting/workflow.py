"""Draft lifecycle + audit log (SQLite). Nothing here sends anything; "approved" only records that a
human signed the text off.

Every action writes one append-only audit row:
    ts, inquiry_id, action, original_draft (what the system generated), edited_draft (the text at
    the time of the action), reason (optional, mainly for rejections).
Actions: generate_draft, save_draft, approve, reject  (+ correct_field from repository.save_correction).
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from typing import Mapping, Optional

from src.database.repository import iso
from src.drafting.draft import DraftContext, approval_blockers, find_violations

FINAL_STATUSES = ("approved", "rejected")


class DraftError(ValueError):
    """An action that is not allowed (empty text, a finalised draft, an outdated draft, banned content)."""


def checklist_fingerprint(statuses: Mapping[str, str]) -> str:
    """Short hash of the checklist STATUSES (which fields are provided / missing / unclear / n.a.).

    A draft asks for exactly the missing/unclear fields, so any status change makes an existing draft
    outdated. Value-only edits do not change what a draft asks and are deliberately ignored.
    """
    canon = "|".join(f"{k}={statuses[k]}" for k in sorted(statuses))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def is_stale(draft: sqlite3.Row, statuses: Mapping[str, str]) -> bool:
    """True if the checklist changed since the draft was written (an unknown fingerprint counts as changed)."""
    return not draft["checklist_hash"] or draft["checklist_hash"] != checklist_fingerprint(statuses)


def _audit(conn: sqlite3.Connection, ts: datetime, inquiry_id: int, action: str,
           original: Optional[str], edited: Optional[str], reason: Optional[str]) -> None:
    conn.execute(
        "INSERT INTO audit_log (ts, inquiry_id, action, original_draft, edited_draft, reason) VALUES (?,?,?,?,?,?)",
        (iso(ts), inquiry_id, action, original, edited, reason))


def create_draft(conn: sqlite3.Connection, inquiry_id: int, body: str, generated_by: str,
                 ts: datetime, note: Optional[str] = None, checklist_hash: str = "") -> int:
    """Store a newly generated draft (status 'draft') with the checklist fingerprint it was written for, and log it."""
    cur = conn.execute(
        """INSERT INTO drafts (inquiry_id, original_body, body, status, generated_by, note, checklist_hash,
                               created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (inquiry_id, body, body, "draft", generated_by, note, checklist_hash, iso(ts), iso(ts)))
    _audit(conn, ts, inquiry_id, "generate_draft", body, None, f"generated_by={generated_by}" + (f"; {note}" if note else ""))
    conn.commit()
    return int(cur.lastrowid)


def get_current_draft(conn: sqlite3.Connection, inquiry_id: int) -> Optional[sqlite3.Row]:
    """The most recently created draft for an inquiry (earlier ones stay in the table for history)."""
    return conn.execute("SELECT * FROM drafts WHERE inquiry_id = ? ORDER BY id DESC LIMIT 1",
                        (inquiry_id,)).fetchone()


def _editable(conn: sqlite3.Connection, draft_id: int, text: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        raise DraftError(f"draft {draft_id} does not exist")
    if row["status"] in FINAL_STATUSES:
        raise DraftError(f"draft is already {row['status']}; generate a new draft to change it")
    if not (text or "").strip():
        raise DraftError("the draft text is empty")
    return row


def _update(conn: sqlite3.Connection, row: sqlite3.Row, status: str, text: str, ts: datetime,
            action: str, reason: Optional[str]) -> None:
    conn.execute("UPDATE drafts SET body = ?, status = ?, updated_at = ? WHERE id = ?",
                 (text, status, iso(ts), row["id"]))
    _audit(conn, ts, row["inquiry_id"], action, row["original_body"], text, reason)
    conn.commit()


def save_draft(conn: sqlite3.Connection, draft_id: int, text: str, ts: datetime) -> None:
    row = _editable(conn, draft_id, text)
    _update(conn, row, "draft", text, ts, "save_draft", None)


def approve_draft(conn: sqlite3.Connection, draft_id: int, text: str, ts: datetime, ctx: DraftContext) -> None:
    """Human sign-off, enforced here (not just in the UI). Refused, with NO audit row, when:
      * the checklist changed since the draft was written (generate a new draft for the current state), or
      * the text breaks a hard content rule (prices, premiums, promises, eligibility, time commitments,
        dates the customer did not write).
    Problematic text can still be saved or rejected. The approved text (with edits) is stored; the
    generated original stays in original_body."""
    row = _editable(conn, draft_id, text)
    if is_stale(row, ctx.statuses):
        raise DraftError("The checklist has changed since this draft was written, so it cannot be approved. "
                         "Generate a new draft for the current checklist.")
    blockers = approval_blockers(find_violations(text, ctx.message, [f.label for f in ctx.provided_fields()]))
    if blockers:
        raise DraftError("This text cannot be approved: " + "; ".join(blockers) +
                         ". Edit it (or reject the draft).")
    _update(conn, row, "approved", text, ts, "approve", None)


def reject_draft(conn: sqlite3.Connection, draft_id: int, text: str, ts: datetime,
                 reason: Optional[str] = None) -> None:
    row = _editable(conn, draft_id, text)
    _update(conn, row, "rejected", text, ts, "reject", (reason or "").strip() or None)


def audit_history(conn: sqlite3.Connection, inquiry_id: int) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM audit_log WHERE inquiry_id = ? ORDER BY id", (inquiry_id,)).fetchall()
