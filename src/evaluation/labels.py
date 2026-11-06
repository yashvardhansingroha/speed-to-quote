"""Read and validate the labels YOU wrote (eval/eval_labeled.csv).

Nothing here creates or suggests a label. If the file is missing, or any label cell is empty or invalid, the
state is reported as "awaiting labels" and no metric is computed.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import Checklist
from src.evaluation.export import label_columns
from src.extraction.schema import INTENTS, STATUSES, URGENCIES

TOTAL_EXPECTED = 80


@dataclass
class LabelState:
    state: str                                  # "no_file" | "incomplete" | "invalid" | "ready"
    n_rows: int = 0
    n_complete: int = 0                         # rows with every label cell filled with a valid value
    errors: list[str] = field(default_factory=list)
    labels: dict[str, dict[str, str]] = field(default_factory=dict)   # eval_id -> {column: value}

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def message(self) -> str:
        if self.state == "no_file":
            return "Awaiting labels: eval/eval_labeled.csv was not found."
        if self.state == "ready":
            return f"Labels loaded: {self.n_complete} of {self.n_rows} rows."
        if self.state == "invalid":
            return f"Awaiting valid labels: {len(self.errors)} problem(s) found in eval_labeled.csv."
        return f"Awaiting labels: {self.n_complete} of {self.n_rows} rows are fully labelled."


def _norm(value: Optional[str]) -> str:
    return re.sub(r"[\s-]+", "_", (value or "").strip().lower())


class LabelFileError(ValueError):
    """The labelled file cannot be read as a CSV table (binary, empty, or an unknown text encoding)."""


def _decode(raw: bytes) -> str:
    """UTF-8 (with or without BOM) first; then Windows-1252, which is what Excel's plain "CSV" save often writes."""
    if b"\x00" in raw:
        raise LabelFileError("the file contains binary data, not text: save it as CSV from your spreadsheet program")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise LabelFileError("the file's text encoding is not recognised: save it as 'CSV UTF-8' from your spreadsheet program")


def _delimiter(first_line: str) -> str:
    """Excel in some regions writes ';' (or tabs) instead of ','. Pick whichever the header row uses most."""
    counts = {d: first_line.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = _decode(Path(path).read_bytes())
    if not text.strip():
        raise LabelFileError("the file is empty")
    lines = text.splitlines()
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=_delimiter(lines[0]))
    return [h.strip() for h in (reader.fieldnames or [])], [{(k or "").strip(): v for k, v in row.items()} for row in reader]


def load_labels(labeled: Path, unlabeled: Path, checklist: Checklist) -> LabelState:
    """Validate eval_labeled.csv against eval_unlabeled.csv (same rows, every label filled with an allowed value)."""
    if not labeled.exists():
        return LabelState("no_file")
    cols = label_columns(checklist)
    allowed = {"intent": set(INTENTS), "urgency": set(URGENCIES)} | {c: set(STATUSES) for c in cols[2:]}
    try:
        header, rows = read_csv(labeled)
    except LabelFileError as exc:
        return LabelState("invalid", 0, 0, [f"eval_labeled.csv cannot be read: {exc}"])
    errors: list[str] = []
    missing_cols = [c for c in ["eval_id"] + cols if c not in header]
    if missing_cols:
        return LabelState("invalid", len(rows), 0, [f"missing column(s): {', '.join(missing_cols)}"])

    expected_ids = [r["eval_id"] for r in read_csv(unlabeled)[1]] if unlabeled.exists() else []
    ids = [r["eval_id"] for r in rows]
    if expected_ids and set(ids) != set(expected_ids):
        errors.append(f"eval_id values do not match eval_unlabeled.csv "
                      f"({len(set(expected_ids) - set(ids))} missing, {len(set(ids) - set(expected_ids))} unexpected)")
    if len(set(ids)) != len(ids):
        errors.append("duplicate eval_id values")

    labels: dict[str, dict[str, str]] = {}
    n_complete = 0
    for r in rows:
        row_ok, vals = True, {}
        for c in cols:
            raw = (r.get(c) or "").strip()
            if not raw:
                row_ok = False
                continue
            v = _norm(raw)
            if v not in allowed[c]:
                errors.append(f"{r['eval_id']}: {c} = '{raw}' is not one of {sorted(allowed[c])}")
                row_ok = False
            vals[c] = v
        if row_ok:
            n_complete += 1
            labels[r["eval_id"]] = vals
    if errors:
        return LabelState("invalid", len(rows), n_complete, errors)
    if n_complete < len(rows) or not rows:
        return LabelState("incomplete", len(rows), n_complete)
    return LabelState("ready", len(rows), n_complete, [], labels)
