"""Export the 80 evaluation inquiries WITHOUT labels.

    python -m src.evaluation.export

Writes eval/eval_unlabeled.csv. The label columns (intent, urgency and one status column per checklist field)
are left EMPTY on purpose: you label them yourself and save the result as eval/eval_labeled.csv. This code never
pre-fills, suggests or generates a label. The generator's internal scenario information (what kind of message it
meant to write) is discarded here and is not written anywhere.

The evaluation messages are separate from the operational data: a different random seed, and any message that
also appears in the operational set is dropped.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Optional

from src.config import ROOT, Checklist, load_checklist, load_settings, reference_now
from src.data_generation.messages import generate_inquiries

EVAL_DIR_ENV = "STQ_EVAL_DIR"


def eval_dir() -> Path:
    """eval/ by default; overridable (tests and the UI tests use a temp folder)."""
    return Path(os.environ.get(EVAL_DIR_ENV) or ROOT / "eval")


def unlabeled_path() -> Path:
    return eval_dir() / "eval_unlabeled.csv"


def labeled_path() -> Path:
    return eval_dir() / "eval_labeled.csv"


def cache_path() -> Path:
    return eval_dir() / "llm_cache.json"


def label_columns(checklist: Checklist) -> list[str]:
    return ["intent", "urgency"] + [f"status_{k}" for k in checklist.keys]


def csv_columns(checklist: Checklist) -> list[str]:
    return ["eval_id", "received_date", "message"] + label_columns(checklist)


def generate_eval_messages(count: int, seed: int, operational_messages: set[str]) -> list[dict]:
    """`count` unique messages that do not appear in the operational data, in shuffled order."""
    s = load_settings()
    now = reference_now()
    seen = set(operational_messages)
    out: list[dict] = []
    batch = count * 2
    while len(out) < count:
        for g in generate_inquiries(batch, seed, now, s["synthetic"]["history_days"]):
            if g.message not in seen:
                seen.add(g.message)
                out.append({"received_at": g.received_at, "message": g.message})
        batch *= 2                                        # (rarely needed) widen the pool and try again
    out = out[:count]
    random.Random(seed).shuffle(out)                      # no ordering that could hint at anything
    return out


def export_unlabeled(path: Optional[Path] = None, checklist: Optional[Checklist] = None) -> Path:
    checklist = checklist or load_checklist()
    path = path or unlabeled_path()
    s = load_settings()["synthetic"]
    operational = {g.message for g in generate_inquiries(
        s["operational_count"], s["operational_seed"], reference_now(), s["history_days"])}
    rows = generate_eval_messages(s["eval_count"], s["eval_seed"], operational)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = csv_columns(checklist)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:       # utf-8-sig: opens cleanly in Excel
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"eval_id": f"E{i:03d}", "received_date": f"{r['received_at']:%Y-%m-%d}",
                        "message": r["message"], **{c: "" for c in label_columns(checklist)}})
    return path


if __name__ == "__main__":
    p = export_unlabeled()
    print(f"Wrote {p}. Label columns are empty: fill them in and save as {labeled_path()}")
