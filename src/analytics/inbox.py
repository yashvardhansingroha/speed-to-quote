"""Priority inbox: currently OPEN inquiries ranked by the explainable priority score.

Inquiries that were routed to human review (unclear intent / too short / unreliable urgency quote) have no
score, so they are listed separately instead of being ranked.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.analysis import Analysis, load_open_analyses
from src.database import repository as repo
from src.scoring.priority import followup_status


@dataclass
class Inbox:
    ranked: pd.DataFrame        # scored open inquiries, best first
    review: pd.DataFrame        # open inquiries that need a human to look first (no score)
    analyses: dict[int, Analysis]


def build_inbox(conn: sqlite3.Connection, now: datetime) -> Inbox:
    analyses, counts = load_open_analyses(conn, now)
    due: dict[int, list[datetime]] = defaultdict(list)
    for r in conn.execute("SELECT inquiry_id, due_at FROM followup_tasks WHERE status = 'open'"):
        due[r["inquiry_id"]].append(repo.parse_ts(r["due_at"]))
    ranked_rows, review_rows = [], []
    for iid, a in analyses.items():
        provided, applicable = counts.get(iid, (0, 0))
        base = {
            "Inquiry": iid, "Intent": a.extraction.intent, "Stated urgency": a.extraction.urgency,
            "Completeness": f"{provided} of {applicable}",
            "Follow-up": followup_status(due.get(iid, []), now), "Received": a.inquiry["received_at"].replace("T", " ")[:16],
            "Message": " ".join(a.inquiry["message"].split())[:90],
        }
        tri = a.triage
        if tri.needs_review:
            review_rows.append({**base, "Why": " ".join(tri.review_reasons)})
        else:
            days = tri.days_until_deadline
            ranked_rows.append({
                **base, "Score": tri.priority.score,
                "Deadline (days)": days if days is not None else None,
                "Waiting (h)": round(tri.hours_waiting, 1) if tri.hours_waiting is not None else None,
            })
    ranked = pd.DataFrame(ranked_rows)
    if not ranked.empty:
        ranked = ranked.sort_values(["Score", "Inquiry"], ascending=[False, True]).reset_index(drop=True)
        ranked.insert(0, "Rank", ranked.index + 1)
    return Inbox(ranked, pd.DataFrame(review_rows), analyses)
