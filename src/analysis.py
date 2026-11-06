"""Glue between extraction, triage and the database (a thin service layer for a user interface or a script).

`analyze_inquiry` runs the pipeline (LLM if available, rules otherwise), stores the result and
returns what the UI needs. `load_analysis` rebuilds the same view from what is already stored, with
no API call, so simply browsing the inbox never spends quota.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.config import Checklist
from src.database import repository as repo
from src.drafting.draft import DraftContext
from src.extraction.llm_client import LLMClient
from src.extraction.pipeline import run_extraction
from src.extraction.schema import InquiryExtraction
from src.scoring.priority import followup_status
from src.scoring.triage import TriageResult, triage_inquiry


@dataclass
class Analysis:
    inquiry: sqlite3.Row
    extraction: InquiryExtraction
    method: str                       # "llm" | "rules"
    model: Optional[str]
    fallback_reason: Optional[str]
    triage: TriageResult


def _triage(conn: sqlite3.Connection, inq: sqlite3.Row, ex: InquiryExtraction, now: datetime) -> TriageResult:
    due = repo.open_task_due_times(conn, inq["id"])
    return triage_inquiry(
        inq["message"], ex, repo.parse_ts(inq["received_at"]), now,
        first_response_at=repo.first_response_at(conn, inq["id"]),
        followup=followup_status(due, now))


def analyze_inquiry(conn: sqlite3.Connection, inquiry_id: int, checklist: Checklist,
                    client: Optional[LLMClient], now: datetime) -> Analysis:
    """Run extraction now (LLM or rules), store extraction + checklist, and triage."""
    inq = repo.get_inquiry(conn, inquiry_id)
    received = repo.parse_ts(inq["received_at"])
    outcome = run_extraction(inq["message"], checklist, received, client)
    tri = _triage(conn, inq, outcome.extraction, now)
    repo.save_extraction(
        conn, inquiry_id, outcome, tri.needs_review, " | ".join(tri.review_reasons) or None,
        tri.deadline_date.isoformat() if tri.deadline_date else None, now,
        tri.deadline_latest.isoformat() if tri.deadline_latest else None, tri.deadline_approximate)
    repo.save_checklist(conn, inquiry_id, outcome, now, keep_manual=True)
    conn.commit()
    return Analysis(inq, outcome.extraction, outcome.method, outcome.model, outcome.fallback_reason, tri)


def load_analysis(conn: sqlite3.Connection, inquiry_id: int, now: datetime) -> Optional[Analysis]:
    """Rebuild the analysis from stored data (no extraction call). None if never analysed."""
    inq = repo.get_inquiry(conn, inquiry_id)
    stored = repo.latest_extraction(conn, inquiry_id)
    if inq is None or stored is None:
        return None
    row, ex = stored
    return Analysis(inq, ex, row["method"], row["model"], row["fallback_reason"], _triage(conn, inq, ex, now))


def draft_context(conn: sqlite3.Connection, analysis: Analysis, checklist: Checklist) -> DraftContext:
    """Collect what a follow-up draft may use: the CURRENT checklist (incl. human corrections)."""
    rows = repo.checklist_rows(conn, analysis.inquiry["id"])
    inq = analysis.inquiry
    first = inq["customer_name"].split()[0] if inq["channel"] != "pasted" and inq["customer_name"] else None
    ex = analysis.extraction
    words = ex.deadline.value if ex.deadline.status == "provided" else None
    return DraftContext(
        message=inq["message"], customer_first_name=first, checklist=checklist,
        statuses={k: rows[k]["status"] for k in checklist.keys},
        evidence={k: rows[k]["evidence"] for k in checklist.keys},
        evidence_verified={k: (None if rows[k]["evidence_verified"] is None else bool(rows[k]["evidence_verified"]))
                           for k in checklist.keys},
        deadline_words=words, deadline_approximate=analysis.triage.deadline_approximate)


def load_open_analyses(conn: sqlite3.Connection, now: datetime) -> tuple[dict[int, Analysis], dict[int, tuple[int, int]]]:
    """Analyses for ALL open inquiries using a handful of bulk queries (not ~12 queries per inquiry).

    Returns ({inquiry_id: Analysis}, {inquiry_id: (fields provided, fields applicable)}). The numbers are identical to
    calling load_analysis() one inquiry at a time (a test checks this); it is just much faster for the priority inbox.
    """
    inquiries = {r["id"]: r for r in conn.execute("SELECT * FROM inquiries WHERE status = 'open' ORDER BY id")}
    latest = {r["inquiry_id"]: r for r in conn.execute(
        "SELECT e.* FROM extractions e JOIN (SELECT inquiry_id, MAX(id) AS mid FROM extractions GROUP BY inquiry_id) m "
        "ON e.id = m.mid")}
    first = {r["inquiry_id"]: repo.parse_ts(r["t"]) for r in conn.execute(
        "SELECT inquiry_id, MIN(occurred_at) AS t FROM events WHERE event_type = 'first_response' GROUP BY inquiry_id")}
    due: dict[int, list[datetime]] = defaultdict(list)
    for r in conn.execute("SELECT inquiry_id, due_at FROM followup_tasks WHERE status = 'open'"):
        due[r["inquiry_id"]].append(repo.parse_ts(r["due_at"]))
    counts = {r["inquiry_id"]: (int(r["provided"] or 0), int(r["applicable"] or 0)) for r in conn.execute(
        "SELECT inquiry_id, SUM(status = 'provided') AS provided, SUM(status != 'not_applicable') AS applicable "
        "FROM checklist_status GROUP BY inquiry_id")}

    out: dict[int, Analysis] = {}
    for iid, inq in inquiries.items():
        row = latest.get(iid)
        if row is None:
            continue
        ex = InquiryExtraction.model_validate_json(row["raw_json"])
        tri = triage_inquiry(inq["message"], ex, repo.parse_ts(inq["received_at"]), now,
                             first_response_at=first.get(iid), followup=followup_status(due.get(iid, []), now))
        out[iid] = Analysis(inq, ex, row["method"], row["model"], row["fallback_reason"], tri)
    return out, counts
