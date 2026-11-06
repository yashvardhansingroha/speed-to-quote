"""Date-first deadlines, value support for extracted fields, and urgency context."""
from datetime import date, datetime

import pytest

from src.extraction.deadlines import find_deadline
from src.extraction.evidence import verify_extraction
from src.extraction.rules import extract_with_rules
from src.extraction.schema import validate_llm_payload
from src.scoring.triage import triage_inquiry
from tests.conftest import llm_payload

RECEIVED = datetime(2026, 11, 2, 9, 0)


# ---- 1. a deadline word AFTER the date counts too -------------------------------------------------
def test_date_first_deadline_is_found():
    hit = find_deadline("We need a new insurance quote. November 6 is our policy expiry date.", RECEIVED)
    assert hit is not None and hit.date == date(2026, 11, 6)


def test_date_first_deadline_gets_scored(checklist, now):
    text = "We need a new insurance quote. November 6 is our policy expiry date."
    ex = extract_with_rules(text, checklist, now)
    tri = triage_inquiry(text, ex, now, now)
    assert ex.deadline.status == "provided" and tri.days_until_deadline == 4
    assert tri.priority.factors[1].points == 40


@pytest.mark.parametrize("text", [
    "March 3, 2020 is when we bought our first truck.",      # trailing words are history, not a deadline
    "June 12 was the day we started the business.",
    "We bought 3 trucks on March 3, 2020. We need a new insurance quote.",
])
def test_date_first_history_is_still_not_a_deadline(text):
    assert find_deadline(text, RECEIVED) is None


def test_cue_in_the_next_sentence_does_not_leak_back():
    assert find_deadline("We started on November 6. Our policy expires someday.", RECEIVED) is None


# ---- 2a. a fleet count must be tied to vehicle wording ------------------------------------------
def _verified(checklist, text, **fields):
    p = llm_payload(checklist)
    p["fields"].update(fields)
    return verify_extraction(validate_llm_payload(p, checklist.keys), text, checklist)


def test_fleet_size_cannot_come_from_the_driver_count(checklist):
    text = "We have 12 trucks and have 99 drivers"
    ex = _verified(checklist, text, fleet_size={"status": "provided", "value": "99", "evidence": text})
    assert ex.fields["fleet_size"].status == "unclear" and ex.fields["fleet_size"].value is None


@pytest.mark.parametrize("text,evidence,value", [
    ("We have 12 trucks and have 99 drivers", "12 trucks", "12"),
    ("We have twelve power units", "twelve power units", "12"),
    ("Our fleet of 7 is growing", "fleet of 7", "7"),
    ("just me and one truck", "one truck", "1"),
])
def test_fleet_size_tied_to_vehicles_is_accepted(checklist, text, evidence, value):
    ex = _verified(checklist, text, fleet_size={"status": "provided", "value": value, "evidence": evidence})
    assert ex.fields["fleet_size"].status == "provided"


def test_rules_baseline_does_not_take_driver_count_as_fleet(checklist, now):
    ex = extract_with_rules("We have 99 drivers and need a quote for them", checklist, now)
    assert ex.fields["fleet_size"].status == "missing"


# ---- 2b. 'not applicable' needs a quote that actually shows non-applicability -----------------
def test_loss_runs_not_applicable_needs_new_venture_wording(checklist):
    ex = _verified(checklist, "We have 12 trucks",
                   loss_runs={"status": "not_applicable", "value": None, "evidence": "12 trucks"})
    f = ex.fields["loss_runs"]
    assert f.status == "unclear" and "not applicable" in f.note.lower()


@pytest.mark.parametrize("quote", [
    "just got our authority", "brand new company", "no prior insurance", "first truck",
])
def test_loss_runs_not_applicable_with_real_new_venture_wording(checklist, quote):
    ex = _verified(checklist, f"We {quote} and need a quote",
                   loss_runs={"status": "not_applicable", "value": None, "evidence": quote})
    assert ex.fields["loss_runs"].status == "not_applicable"


# ---- 3. historical wording must not create urgency ---------------------------------------------
@pytest.mark.parametrize("text", [
    "We need a quote. We bought another truck today.",
    "We need a quote for our fleet. We hired two drivers yesterday and a third this week.",
    "I called you on Tuesday. We need a quote for 5 trucks, whenever works.",
])
def test_history_words_do_not_make_high_urgency(checklist, now, text):
    assert extract_with_rules(text, checklist, now).urgency != "high"


@pytest.mark.parametrize("text", [
    "We need a quote today.", "Can you call me tomorrow about a quote?", "We need coverage this week.",
    "Policy lapsed, I need insurance, this is urgent.",
])
def test_genuine_time_pressure_still_gives_high_urgency(checklist, now, text):
    assert extract_with_rules(text, checklist, now).urgency == "high"


def test_urgent_after_a_past_event_still_counts(checklist, now):
    """'urgent' is not a time word, so a past verb earlier in the sentence must not cancel it."""
    assert extract_with_rules("I called yesterday and it's urgent, we need a quote", checklist, now).urgency == "high"


def test_original_spec_example_still_scores_65(checklist, now):
    text = ("Hi, we run a trucking business with 12 trucks operating in Texas and Oklahoma. Our insurance "
            "expires next month. We're looking for a new quote and want to talk to someone this week.")
    ex = extract_with_rules(text, checklist, now)
    tri = triage_inquiry(text, ex, now, now)
    # earlier request ("this week", Fri 6 Nov) wins over the vaguer "next month" expiry
    assert tri.priority.score == 65 and tri.days_until_deadline == 4 and not tri.deadline_approximate
