"""Deterministic evidence checking - the guard against invented values.

Rules enforced here (all in code, none left to the model):
  1. A quote must appear VERBATIM in the message (only runs of whitespace are collapsed).
  2. A `provided` field needs a quote; `missing` carries no value.
  3. The value must be supported by its quote for numeric / state / ID fields
     (a real quote "12 trucks" cannot justify a value of 99).
  4. `not_applicable` needs a quote, and only fields that can be non-applicable in the config.
  5. A classification (intent) whose supporting quote is invented becomes `unclear`
     (-> human review); an invented urgency quote is flagged so triage routes it to review.
"""
from __future__ import annotations

import re
from typing import Optional

from src.config import Checklist, ChecklistField, load_triage_rules
from src.extraction.deadlines import is_deadline_context
from src.extraction.schema import NOT_PROVIDED, FieldExtraction, InquiryExtraction
from src.extraction.textutils import (
    FLEET_RX, NUM_PATTERN, STATE_ABBRS, STATE_NAMES, collapse_ws, parse_number,
)
from src.extraction.patterns import compile_patterns

MIN_QUOTE_CHARS = 2


def evidence_in_text(evidence: str, source: str) -> bool:
    """True if `evidence` occurs verbatim in `source` (case-sensitive; whitespace runs collapsed)."""
    if not evidence or evidence.strip() == NOT_PROVIDED:
        return False
    ev = collapse_ws(evidence)
    if len(ev) < MIN_QUOTE_CHARS:
        return False
    return ev in collapse_ws(source)


# --------------------------------------------------------------------------- value support checks
_NUM_RX = re.compile(rf"\b{NUM_PATTERN}\b", re.IGNORECASE)


def _numbers_in(text: str) -> set[int]:
    return {n for m in _NUM_RX.finditer(text) if (n := parse_number(m.group(0))) is not None}


def _states_in(text: str) -> set[str]:
    found = {ab for ab in re.findall(r"(?<![\w.])[A-Z]{2}(?!\w)", text) if ab in STATE_ABBRS}
    low = text.lower()
    found |= {ab for name, ab in STATE_NAMES.items() if re.search(rf"\b{name}\b", low)}
    return found


def _fleet_counts_in(text: str) -> set[int]:
    """Numbers that are tied to vehicle wording (so '99 drivers' is NOT a fleet count)."""
    out = set()
    for m in FLEET_RX.finditer(text):
        n = parse_number(m.group("n") or m.group("n2"))
        if n:
            out.add(n)
    return out


def value_supported(kind: Optional[str], value: Optional[str], evidence: str) -> bool:
    """Is `value` actually backed by `evidence`? Unknown/None kind => no extra check."""
    if kind is None:
        return True
    if not value:
        return False
    if kind == "integer":
        n = parse_number(value)
        return n is not None and n > 0 and n in _numbers_in(evidence)
    if kind == "fleet_count":
        n = parse_number(value)
        return n is not None and n > 0 and n in _fleet_counts_in(evidence)
    if kind == "states":
        wanted = set()
        for tok in re.split(r"[,/&\s]+|\band\b", value):
            tok = tok.strip()
            if tok:
                wanted.add(tok.upper() if tok.upper() in STATE_ABBRS else STATE_NAMES.get(tok.lower(), tok))
        return bool(wanted) and wanted <= _states_in(evidence)
    if kind == "digits":
        runs = re.findall(r"\d{5,}", value.replace(",", ""))
        ev_digits = set(re.findall(r"\d{5,}", evidence.replace(",", "")))
        return bool(runs) and set(runs) <= ev_digits
    return True


def urgency_quote_is_historical(quote: str, source: str) -> bool:
    """True if the urgency quote only 'signals urgency' through time words used as history.

    "We bought another truck today" contains "today", but it is not a request for speed. If every
    urgency cue inside the quote is a context-sensitive time word whose surrounding sentence
    reads as past, the quote does not support an urgency label.
    """
    rules = load_triage_rules()["urgency_rules"]
    time_rx = re.compile(rules["context_sensitive"], re.IGNORECASE)
    cues = [m for p in rules["high"] + rules["medium"] for m in re.finditer(p, quote, re.IGNORECASE)]
    if not cues or not all(time_rx.fullmatch(m.group(0)) for m in cues):
        return False                                   # a non-time cue ("urgent", "ASAP") backs it up
    q = collapse_ws(quote)
    base = collapse_ws(source).find(q)
    if base < 0:
        return False
    src = collapse_ws(source)
    return all(not is_deadline_context(src, base + m.start(), base + m.end()) for m in cues)


# --------------------------------------------------------------------------- per-field check
_NA_CACHE: dict[tuple, list] = {}


def _na_supported(spec: ChecklistField, evidence: str, macros: dict[str, str]) -> bool:
    """A 'not applicable' quote must match one of the field's configured non-applicability patterns."""
    key = (spec.key, tuple(spec.rules.na_patterns))
    if key not in _NA_CACHE:
        _NA_CACHE[key] = compile_patterns(spec.rules.na_patterns, macros)
    return any(rx.search(evidence) for rx in _NA_CACHE[key])


def _check_field(f: FieldExtraction, source: str, spec: Optional[ChecklistField] = None,
                 macros: Optional[dict[str, str]] = None) -> FieldExtraction:
    f = f.model_copy()
    if f.status == "missing":
        f.value, f.evidence, f.evidence_verified = None, NOT_PROVIDED, None
        return f

    def downgrade(note: str) -> FieldExtraction:
        f.status, f.value, f.note = "unclear", None, note
        return f

    has_quote = f.evidence != NOT_PROVIDED
    if not has_quote:
        f.evidence_verified = None
        if f.status == "provided":
            return downgrade("Marked provided without a supporting quote; needs human check.")
        if f.status == "not_applicable":
            return downgrade("Marked not applicable without a supporting quote; needs human check.")
        return f

    if not evidence_in_text(f.evidence, source):
        claimed = f.value
        f.evidence_verified = False
        return downgrade("Quote not found in the message" + (f" (model claimed '{claimed}')." if claimed else "."))

    f.evidence_verified = True
    if f.status == "not_applicable" and spec is not None:
        if not spec.rules.na_patterns:
            return downgrade("This field cannot be 'not applicable' for this line; needs human check.")
        if not _na_supported(spec, f.evidence, macros or {}):
            return downgrade("The quote does not show that this field is not applicable "
                             "(e.g. no statement that this is a new venture); needs human check.")
    if f.status == "provided":
        if not f.value:
            f.value = f.evidence
        elif spec is not None and not value_supported(spec.value_check, f.value, f.evidence):
            claimed = f.value
            return downgrade(f"Value '{claimed}' is not supported by the quoted text.")
    return f


def verify_extraction(ex: InquiryExtraction, source: str,
                      checklist: Optional[Checklist] = None) -> InquiryExtraction:
    """Verify every quote and value in an extraction. Returns a new, corrected object."""
    out = ex.model_copy(deep=True)
    specs = {f.key: f for f in checklist.fields} if checklist else {}
    out.business_description = _check_field(out.business_description, source)
    out.deadline = _check_field(out.deadline, source)
    out.fields = {k: _check_field(v, source, specs.get(k), checklist.macros if checklist else None) for k, v in out.fields.items()}

    # Classifications: a quote is optional ("not_provided" is allowed) but, if given, it must be real.
    out.intent_evidence_verified = (
        None if out.intent_evidence == NOT_PROVIDED else evidence_in_text(out.intent_evidence, source))
    out.urgency_evidence_verified = (
        None if out.urgency_evidence == NOT_PROVIDED else evidence_in_text(out.urgency_evidence, source))
    if (out.urgency_evidence_verified and out.urgency in ("high", "medium")
            and urgency_quote_is_historical(out.urgency_evidence, source)):
        out.urgency_evidence_verified = False          # real words, but a past event, not a timing request
    if out.intent_evidence_verified is False and out.intent != "unclear":
        out.intent_note = (f"Model chose '{out.intent}' but its supporting quote is not in the message; "
                           "intent set to 'unclear' for human review.")
        out.intent = "unclear"
    return out
