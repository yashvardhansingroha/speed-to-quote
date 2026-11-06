"""Build the SYNTHETIC demo database.

    python -m src.data_generation.build_database            # 250 inquiries, seed from settings.yaml
    python -m src.data_generation.build_database --n 300 --seed 5

Deterministic: the same arguments always produce the same database content. Extraction here
uses the OFFLINE rules baseline only (no API calls), so building needs no key and is free.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import Checklist, load_checklist, load_settings, reference_now
from src.data_generation.events import simulate_history
from src.data_generation.messages import generate_inquiries
from src.database import db
from src.database.repository import iso, save_checklist, save_extraction
from src.extraction.pipeline import run_extraction
from src.scoring.triage import triage_inquiry


def build_database(db_path: Path = db.DEFAULT_DB_PATH, n: Optional[int] = None, seed: Optional[int] = None,
                   now: Optional[datetime] = None, checklist: Optional[Checklist] = None) -> dict:
    """Create (or recreate) the database and fill it. Returns a small summary dict."""
    settings = load_settings()
    n = n if n is not None else settings["synthetic"]["operational_count"]
    seed = seed if seed is not None else settings["synthetic"]["operational_seed"]
    now = now or reference_now()
    checklist = checklist or load_checklist()

    conn = db.connect(db_path)
    db.drop_all(conn)
    db.init_db(conn)

    inquiries = generate_inquiries(n, seed, now, settings["synthetic"]["history_days"])
    history_rng = random.Random(seed + 1)          # separate stream: history never perturbs the text

    for g in inquiries:
        cur = conn.execute(
            """INSERT INTO inquiries (customer_name, contact_email, contact_phone, channel, message,
                   received_at, status, is_synthetic) VALUES (?,?,?,?,?,?,?,1)""",
            (g.customer_name, g.email, g.phone, g.channel, g.message, iso(g.received_at), "open"))
        inquiry_id = int(cur.lastrowid)

        # Offline extraction (rules). The LLM is only used interactively in the app and the evaluation.
        outcome = run_extraction(g.message, checklist, g.received_at, client=None)
        ex = outcome.extraction
        # Real triage result (not a placeholder): human-review routing is persisted as computed.
        tri = triage_inquiry(g.message, ex, g.received_at, now)
        save_extraction(conn, inquiry_id, outcome, needs_review=tri.needs_review,
                        review_reason=" | ".join(tri.review_reasons) or None,
                        deadline_date=tri.deadline_date.isoformat() if tri.deadline_date else None,
                        created_at=g.received_at,
                        deadline_latest=tri.deadline_latest.isoformat() if tri.deadline_latest else None,
                        deadline_approximate=tri.deadline_approximate)
        save_checklist(conn, inquiry_id, outcome, g.received_at)

        n_provided = sum(1 for f in ex.fields.values() if f.status == "provided")
        hist = simulate_history(history_rng, g.received_at, g.scenario_intent, g.scenario_urgency, n_provided, now)
        for e in hist.events:
            conn.execute("INSERT INTO events (inquiry_id, event_type, occurred_at, detail) VALUES (?,?,?,?)",
                         (inquiry_id, e.event_type, iso(e.occurred_at), e.detail))
        for t in hist.tasks:
            conn.execute(
                """INSERT INTO followup_tasks (inquiry_id, task_type, created_at, due_at, status, completed_at)
                   VALUES (?,?,?,?,?,?)""",
                (inquiry_id, t.task_type, iso(t.created_at), iso(t.due_at), t.status, iso(t.completed_at)))
        conn.execute("UPDATE inquiries SET status = ? WHERE id = ?", (hist.status, inquiry_id))

    conn.commit()
    summary = {
        "db_path": str(db_path), "inquiries": n, "seed": seed, "reference_now": iso(now),
        "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        "tasks": conn.execute("SELECT COUNT(*) FROM followup_tasks").fetchone()[0],
    }
    conn.close()
    return summary


def ensure_database(db_path: Path = db.DEFAULT_DB_PATH) -> Path:
    """Build the demo DB if it is missing or too old to migrate (for example on the first start of a hosted app)."""
    if not db.schema_is_current(db_path) and not db.migrate(db_path):   # migrate keeps user data; rebuild only if needed
        build_database(db_path)
    return Path(db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the SYNTHETIC demo database")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DB_PATH)
    args = parser.parse_args()
    print(build_database(args.db, args.n, args.seed))
