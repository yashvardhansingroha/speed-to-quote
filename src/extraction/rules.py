"""Offline rules-based baseline.

This is both (a) the fallback used whenever the LLM is unavailable or returns garbage and
(b) the baseline the LLM is compared against in the evaluation. It is intentionally simple:
regular expressions from /config, nothing learned. Every value it returns is a slice of the
message, so its evidence quotes are verbatim by construction.
"""
from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Optional

from src.config import Checklist, ChecklistField, load_triage_rules
from src.extraction.deadlines import NEGATION_RX, clause_around, find_deadline, is_deadline_context
from src.extraction.evidence import verify_extraction
from src.extraction.patterns import compile_patterns as _compile
from src.extraction.schema import NOT_PROVIDED, FieldExtraction, InquiryExtraction
from src.extraction.textutils import (
    AMBIGUOUS_ABBRS, FLEET_BOUND_RX, FLEET_RANGE_RX, FLEET_RX as _FLEET_RX, STATE_ABBRS, STATE_NAMES, parse_number,
)

# --------------------------------------------------------------------------- pattern helpers
def _first(patterns: list[re.Pattern], text: str) -> Optional[re.Match]:
    """Earliest match in the text across all patterns."""
    hits = [m for rx in patterns if (m := rx.search(text))]
    return min(hits, key=lambda m: m.start()) if hits else None


def _field(status: str, value: Optional[str] = None, evidence: str = NOT_PROVIDED) -> FieldExtraction:
    return FieldExtraction(status=status, value=value, evidence=evidence)


def _quote(m: re.Match) -> str:
    return m.group(0).strip(" ,;:")


# --------------------------------------------------------------------------- special extractors
_STATE_NAME_RX = re.compile(
    r"\b(" + "|".join(sorted(STATE_NAMES, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_STATE_ABBR_RX = re.compile(r"(?<![\w.])([A-Z]{2})(?![\w])")           # case-sensitive on purpose
_STATE_GAP_RX = re.compile(r"^\s*(?:,|/|&|-|\u2013|\u2014|\+|and|or|then|to|also|plus)?\s*(?:and\s+)?$", re.IGNORECASE)
# "Kansas City", "Oklahoma City" and "Washington DC" name a city, not (necessarily) the state.
_CITY_TAIL_RX = re.compile(r"\s+City\b", re.IGNORECASE)
_DC_TAIL_RX = re.compile(r",?\s*D\.?C\b")
_NATIONWIDE_RX = re.compile(r"\b(?:nationwide|all 48 states|lower 48|all states|coast to coast|all over the (?:us|country))\b",
                            re.IGNORECASE)


def _extract_states(text: str) -> FieldExtraction:
    """Find state names/abbreviations and group neighbours into one evidence span.

    Ambiguous abbreviations (OK, IN, OR, ME, ...) only count when they sit right next to another
    state, so "OK thanks" is ignored but "TX and OK" is understood.
    """
    mentions: list[tuple[int, int, str, bool]] = []     # start, end, abbr, ambiguous?
    skipped: list[str] = []                             # city-like mentions: kept as "unclear" evidence if nothing else
    for m in _STATE_NAME_RX.finditer(text):
        tail = text[m.end():m.end() + 8]
        city = _CITY_TAIL_RX.match(tail) or (m.group(1).lower() == "washington" and _DC_TAIL_RX.match(tail))
        if city:
            skipped.append(text[m.start():m.end() + city.end()])
            continue
        mentions.append((m.start(), m.end(), STATE_NAMES[m.group(1).lower()], False))
    for m in _STATE_ABBR_RX.finditer(text):
        ab = m.group(1)
        if ab in STATE_ABBRS:
            mentions.append((m.start(), m.end(), ab, ab in AMBIGUOUS_ABBRS))
    mentions.sort()

    # Group mentions that are separated only by punctuation/"and" (or sit in the same short clause).
    groups: list[list[tuple[int, int, str, bool]]] = []
    for item in mentions:
        if groups:
            prev = groups[-1][-1]
            gap = text[prev[1]:item[0]]
            # Strict neighbours ("TX, OK" / "TX and OK") always join. A looser gap ("TX ... also NM")
            # joins only if neither mention is an ambiguous word like OK/IN/OR.
            loose_ok = (len(gap) <= 40 and not re.search(r"[.!?\n]", gap)
                        and not prev[3] and not item[3])
            if _STATE_GAP_RX.match(gap) or loose_ok:
                groups[-1].append(item)
                continue
        groups.append([item])

    best: Optional[list] = None
    for g in groups:
        solid = [x for x in g if not x[3]]
        if not solid:
            continue                                   # only ambiguous words like "OK" -> ignore
        if best is None or len({x[2] for x in g}) > len({x[2] for x in best}):
            best = g
    if best is None:
        nat = _NATIONWIDE_RX.search(text)
        if nat:
            return _field("unclear", None, _quote(nat))
        if skipped:
            return _field("unclear", None, skipped[0])
        return _field("missing")

    kept = best
    states = []
    for x in kept:
        if x[2] not in states:
            states.append(x[2])
    evidence = text[kept[0][0]:kept[-1][1]]
    return _field("provided", ", ".join(states), evidence)


_FLEET_VAGUE_RX = re.compile(
    r"\b(?:a few|a couple of|a couple|some|several|many|small|growing|handful of)\s+(?:of\s+)?(?:\w+\s+)?"
    r"(?:trucks?|tractors?|units?|fleet|rigs?)\b", re.IGNORECASE)


def _extract_fleet_size(text: str) -> FieldExtraction:
    """Exact count only. A range, a bound ("over 20") or two different counts ("6 dump trucks and 4 flatbeds") is `unclear`,
    so a person confirms it instead of the tool silently reporting one of the numbers."""
    for rng in FLEET_RANGE_RX.finditer(text):
        lo, hi = int(rng.group(1)), int(rng.group(2))
        if lo < hi and len(rng.group(1)) <= 5 and len(rng.group(2)) <= 5:      # "555-0142" (a phone number) is not a range
            return _field("unclear", None, _quote(rng))
    found: list[tuple[int, re.Match]] = []
    for m in _FLEET_RX.finditer(text):
        raw = m.group("n") or m.group("n2")
        n = parse_number(raw)
        if n is None or n == 0:
            continue
        if 1900 <= n <= 2100 and raw.isdigit() and len(raw) == 4:
            continue                                   # a model year, not a count
        window = text[max(0, m.start() - 15):m.start()]
        bound = FLEET_BOUND_RX.search(window)
        if bound:
            start = max(0, m.start() - 15) + bound.start()
            return _field("unclear", None, text[start:m.end()].strip(" ,;:"))
        found.append((n, m))
    if len({n for n, _ in found}) > 1:
        return _field("unclear", None, _quote(found[0][1]))
    if found:
        return _field("provided", str(found[0][0]), _quote(found[0][1]))
    vague = _FLEET_VAGUE_RX.search(text)
    if vague:
        return _field("unclear", None, _quote(vague))
    return _field("missing")


# --------------------------------------------------------------------------- per-field engine
class _FieldMatcher:
    """Compiled regexes for one checklist field."""

    def __init__(self, spec: ChecklistField, macros: dict[str, str]):
        self.spec = spec
        self.patterns = _compile(spec.rules.patterns, macros)
        self.unclear = _compile(spec.rules.unclear_patterns, macros)
        self.na = _compile(spec.rules.na_patterns, macros)

    def run(self, text: str) -> FieldExtraction:
        extractor = self.spec.rules.extractor
        if extractor == "states":
            return _extract_states(text)
        if extractor == "fleet_size":
            return _extract_fleet_size(text)
        # Generic regex field. Order matters: not-applicable, then provided, then unclear.
        if (m := _first(self.na, text)):
            return _field("not_applicable", None, _quote(m))
        if (m := _first(self.patterns, text)):
            return _field("provided", _quote(m), _quote(m))
        if (m := _first(self.unclear, text)):
            return _field("unclear", None, _quote(m))
        return _field("missing")


_MATCHER_CACHE: dict[str, list[_FieldMatcher]] = {}


def _matchers(checklist: Checklist) -> list[_FieldMatcher]:
    """Compile each field's regexes once per checklist (keyed on its content, not identity)."""
    cache_key = checklist.model_dump_json()
    if cache_key not in _MATCHER_CACHE:
        _MATCHER_CACHE[cache_key] = [_FieldMatcher(f, checklist.macros) for f in checklist.fields]
    return _MATCHER_CACHE[cache_key]


@lru_cache(maxsize=1)
def _business_patterns() -> list[re.Pattern]:
    return _compile(load_triage_rules()["business_description_patterns"], {})




# --------------------------------------------------------------------------- intent / urgency
@lru_cache(maxsize=1)
def _intent_rules() -> dict[str, list[tuple[re.Pattern, int]]]:
    raw = load_triage_rules()["intent_rules"]
    return {intent: [(re.compile(r["re"], re.IGNORECASE), int(r["w"])) for r in rules]
            for intent, rules in raw.items()}


def intent_scores(text: str) -> tuple[dict[str, int], dict[str, re.Match]]:
    """Summed rule weight per intent, plus the first matching phrase for each (used by classify_intent and by
    the evaluation's failure diagnosis)."""
    scores: dict[str, int] = {}
    first_hit: dict[str, re.Match] = {}
    for intent, patterns in _intent_rules().items():
        for rx, w in patterns:
            m = rx.search(text)
            if m:
                scores[intent] = scores.get(intent, 0) + w
                if intent not in first_hit or m.start() < first_hit[intent].start():
                    first_hit[intent] = m
    return scores, first_hit


def classify_intent(text: str) -> tuple[str, str]:
    """Return (intent, evidence_quote). Highest summed rule weight wins; no hit => unclear."""
    scores, first_hit = intent_scores(text)
    if not scores:
        return "unclear", NOT_PROVIDED
    order = {name: i for i, name in enumerate(load_triage_rules()["intent_tie_break"])}
    best = max(scores, key=lambda k: (scores[k], -order.get(k, 99)))
    return best, _quote(first_hit[best])


def classify_urgency(text: str, intent: str, days_to_deadline: Optional[int]) -> tuple[str, str]:
    """Return (urgency, evidence_quote)."""
    u = load_triage_rules()["urgency_rules"]

    context_rx = re.compile(u["context_sensitive"], re.IGNORECASE)

    def hit(patterns: list[str], check_negation: bool = True) -> Optional[re.Match]:
        """Earliest match that is not a historical time word or negated ("not urgent", "isn't urgent",
        "don't need it today"). See urgency_rules.context_sensitive."""
        found = []
        for p in patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                if context_rx.fullmatch(m.group(0)) and not is_deadline_context(text, m.start(), m.end()):
                    continue
                if check_negation and NEGATION_RX.search(text[max(0, m.start() - 30):m.start()]):
                    continue
                found.append(m)
                break
        return min(found, key=lambda m: m.start()) if found else None

    if (m := hit(u["high"])):
        return "high", _quote(m)
    if days_to_deadline is not None and days_to_deadline <= u["high_within_days"]:
        return "high", NOT_PROVIDED
    if (m := hit(u["medium"])):
        return "medium", _quote(m)
    if days_to_deadline is not None and days_to_deadline <= u["medium_within_days"]:
        return "medium", NOT_PROVIDED
    if (m := hit(u["low"], check_negation=False)):
        return "low", _quote(m)
    return u["default_urgency"].get(intent, "low"), NOT_PROVIDED


# --------------------------------------------------------------------------- main entry point
def extract_with_rules(text: str, checklist: Checklist, received_at: datetime) -> InquiryExtraction:
    """Run the whole rules baseline on one message and return a verified InquiryExtraction."""
    text = text or ""

    fields = {m.spec.key: m.run(text) for m in _matchers(checklist)}

    # Business description (core field)
    bd = _first(_business_patterns(), text)
    business_description = _field("provided", _quote(bd), _quote(bd)) if bd else _field("missing")

    # Deadline: find the earliest date phrase; evidence is the surrounding clause.
    hit = find_deadline(text, received_at)
    if hit:
        deadline = _field("provided", hit.text, clause_around(text, hit))
        days = (hit.date - received_at.date()).days
    else:
        deadline, days = _field("missing"), None

    intent, intent_ev = classify_intent(text)
    urgency, urgency_ev = classify_urgency(text, intent, days)

    ex = InquiryExtraction(
        intent=intent, urgency=urgency,
        intent_evidence=intent_ev, urgency_evidence=urgency_ev,
        business_description=business_description, deadline=deadline, fields=fields,
    )
    return verify_extraction(ex, text, checklist)
