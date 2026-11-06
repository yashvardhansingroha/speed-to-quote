"""Regression tests for validation, review routing, date handling and evidence checks."""
from datetime import date, datetime

import pytest

from src.data_generation.build_database import build_database
from src.database import db
from src.extraction.deadlines import find_deadline
from src.extraction.evidence import verify_extraction
from src.extraction.pipeline import run_extraction
from src.extraction.rules import extract_with_rules
from src.extraction.schema import validate_llm_payload
from src.scoring.priority import PriorityInputs, compute_priority
from src.scoring.triage import triage_inquiry
from tests.conftest import as_json, llm_payload
from tests.test_pipeline_fallback import StubClient

RECEIVED = datetime(2026, 11, 2, 9, 0)           # a Monday


# ---- 1. human-review flags must be persisted as computed -------------------------------------
def test_needs_review_is_persisted_from_real_triage(built_db, checklist, now):
    rows = built_db.execute(
        "SELECT e.needs_review, e.review_reason, i.message, i.received_at "
        "FROM extractions e JOIN inquiries i ON i.id = e.inquiry_id").fetchall()
    flagged = [r for r in rows if r["needs_review"]]
    assert flagged, "some synthetic messages are too short / ambiguous and must be flagged"
    for r in rows:
        ex = extract_with_rules(r["message"], checklist, datetime.fromisoformat(r["received_at"]))
        tri = triage_inquiry(r["message"], ex, datetime.fromisoformat(r["received_at"]), now)
        assert bool(r["needs_review"]) == tri.needs_review
        assert (r["review_reason"] is not None) == tri.needs_review


# ---- 2. historical dates must not become deadlines -------------------------------------------
@pytest.mark.parametrize("text", [
    "We bought 3 trucks on March 3, 2020. We need a new insurance quote.",
    "We started in March and now have 4 trucks.",
    "I called you on Tuesday about a quote.",
    "We bought the trucks on 3/3 and run 5 units.",
    "Founded in June 2015, we haul gravel.",
])
def test_purchase_and_history_dates_are_not_deadlines(text):
    assert find_deadline(text, RECEIVED) is None


def test_purchase_date_message_gets_no_deadline_points(checklist, now):
    text = "We bought 3 trucks on March 3, 2020. We need a new insurance quote."
    ex = extract_with_rules(text, checklist, now)
    tri = triage_inquiry(text, ex, now, now)
    assert ex.deadline.status == "missing" and tri.priority.factors[1].points == 0
    assert tri.priority.score < 65


def test_dates_long_in_the_past_are_ignored_even_with_a_cue():
    assert find_deadline("our policy expired on March 3, 2020", RECEIVED) is None


def test_real_deadlines_still_work_with_cues():
    assert find_deadline("Our policy expires on March 3, 2027", RECEIVED).date == date(2027, 3, 3)
    assert find_deadline("we need it effective Dec 5", RECEIVED).date == date(2026, 12, 5)


# ---- 3. invented intent / urgency evidence must not produce a score ------------------------
def test_invented_intent_quote_routes_to_review_without_score(checklist, now):
    text = "We have 12 trucks and need a quote please for them"
    p = llm_payload(checklist, intent="new_quote", intent_evidence="made up quote")
    ex = verify_extraction(validate_llm_payload(p, checklist.keys), text, checklist)
    tri = triage_inquiry(text, ex, now, now)
    assert tri.needs_review and tri.priority is None
    assert any("new_quote" in r for r in tri.review_reasons)


def test_invented_urgency_quote_routes_to_review_without_score(checklist, now):
    text = "We have 12 trucks and need a quote please for them"
    p = llm_payload(checklist, urgency="high", urgency_evidence="this is extremely urgent")
    ex = verify_extraction(validate_llm_payload(p, checklist.keys), text, checklist)
    tri = triage_inquiry(text, ex, now, now)
    assert tri.needs_review and tri.priority is None
    assert any("urgency" in r.lower() for r in tri.review_reasons)


def test_real_classification_quotes_are_scored(checklist, now):
    text = "We have 12 trucks and need a quote ASAP please"
    p = llm_payload(checklist, intent_evidence="need a quote", urgency="high", urgency_evidence="ASAP")
    ex = verify_extraction(validate_llm_payload(p, checklist.keys), text, checklist)
    assert not triage_inquiry(text, ex, now, now).needs_review


def test_pipeline_applies_the_same_rule_to_llm_output(checklist, now):
    text = "We run 12 trucks in Texas and our policy expires soon. Please send a quote."
    p = llm_payload(checklist, intent="renewal_or_replacement", intent_evidence="invented words")
    out = run_extraction(text, checklist, now, client=StubClient(answer=as_json(p)))
    assert out.method == "llm" and out.extraction.intent == "unclear"


# ---- 4. a real quote must support the value; N/A needs evidence ----------------------------------
def _verified(checklist, text, **fields):
    p = llm_payload(checklist)
    p["fields"].update(fields)
    return verify_extraction(validate_llm_payload(p, checklist.keys), text, checklist)


def test_value_not_supported_by_its_quote_is_downgraded(checklist):
    ex = _verified(checklist, "We have 12 trucks",
                   fleet_size={"status": "provided", "value": "99", "evidence": "12 trucks"})
    f = ex.fields["fleet_size"]
    assert f.status == "unclear" and f.value is None and "99" in f.note


def test_number_words_support_numeric_values(checklist):
    ex = _verified(checklist, "We run twelve trucks",
                   fleet_size={"status": "provided", "value": "12", "evidence": "twelve trucks"})
    assert ex.fields["fleet_size"].status == "provided"


@pytest.mark.parametrize("value,evidence,ok", [
    ("TX, OK", "Texas and Oklahoma", True), ("TX", "TX/OK", True), ("TX, OK, AR", "Texas and Oklahoma", False),
    ("NM", "Texas and Oklahoma", False),
])
def test_state_values_must_appear_in_the_quote(checklist, value, evidence, ok):
    ex = _verified(checklist, f"We run {evidence} mostly",
                   operating_states={"status": "provided", "value": value, "evidence": evidence})
    assert (ex.fields["operating_states"].status == "provided") is ok


def test_dot_number_must_appear_in_the_quote(checklist):
    good = _verified(checklist, "USDOT 1234567", usdot_mc_number={"status": "provided", "value": "1234567", "evidence": "USDOT 1234567"})
    bad = _verified(checklist, "USDOT 1234567", usdot_mc_number={"status": "provided", "value": "7654321", "evidence": "USDOT 1234567"})
    assert good.fields["usdot_mc_number"].status == "provided" and bad.fields["usdot_mc_number"].status == "unclear"


def test_not_applicable_without_a_quote_becomes_unclear(checklist):
    ex = _verified(checklist, "We need a quote", loss_runs={"status": "not_applicable", "value": None, "evidence": "not_provided"})
    assert ex.fields["loss_runs"].status == "unclear"


def test_not_applicable_is_refused_for_fields_that_cannot_be_na(checklist):
    ex = _verified(checklist, "We have 12 trucks",
                   fleet_size={"status": "not_applicable", "value": None, "evidence": "12 trucks"})
    assert ex.fields["fleet_size"].status == "unclear"


def test_not_applicable_with_a_real_quote_on_an_allowed_field_survives(checklist):
    ex = _verified(checklist, "We just got our authority last week",
                   loss_runs={"status": "not_applicable", "value": None, "evidence": "just got our authority"})
    assert ex.fields["loss_runs"].status == "not_applicable"


# ---- 5. vague dates carry their uncertainty ------------------------------------------------------
def test_next_month_is_a_range_not_a_single_precise_day():
    hit = find_deadline("Our policy expires next month", RECEIVED)
    assert (hit.earliest, hit.latest) == (date(2026, 12, 1), date(2026, 12, 31))
    assert hit.approximate and hit.date == date(2026, 12, 1)


def test_exact_phrases_are_not_marked_approximate():
    assert not find_deadline("policy ends Dec 5", RECEIVED).approximate
    assert not find_deadline("call me this week", RECEIVED).approximate


def test_month_name_alone_is_a_range():
    hit = find_deadline("our policy runs out in March", RECEIVED)
    assert (hit.earliest, hit.latest) == (date(2027, 3, 1), date(2027, 3, 31))


def test_approximate_deadline_is_flagged_in_score_reason_and_hints(checklist, now):
    text = "Our insurance expires next month and we need a quote for 5 trucks."
    ex = extract_with_rules(text, checklist, now)
    tri = triage_inquiry(text, ex, now, now)
    assert tri.deadline_approximate and tri.deadline_latest > tri.deadline_date
    assert "approximate" in tri.priority.factors[1].reason
    assert any("exact date" in h for h in tri.hints)


def test_exact_deadline_has_no_approximation_note():
    r = compute_priority(PriorityInputs("low", 10, "Dec 5"))
    assert "approximate" not in r.factors[1].reason


def test_approximate_deadline_columns_are_stored(built_db):
    row = built_db.execute(
        "SELECT COUNT(*) c FROM extractions WHERE deadline_approximate = 1 AND deadline_latest > deadline_date").fetchone()
    assert row["c"] > 0
