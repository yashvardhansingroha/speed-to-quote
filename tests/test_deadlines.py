"""Deterministic conversion of date phrases to dates.

Reference: Monday 2 Nov 2026, 09:00 (so 'this week' = Fri 6 Nov, etc.).
"""
from datetime import date, datetime

import pytest

from src.extraction.deadlines import clause_around, find_dates, find_deadline, resolve_deadline_date

RECEIVED = datetime(2026, 11, 2, 9, 0)      # a Monday


def d(text):
    hit = find_deadline(text, RECEIVED)
    return hit.date if hit else None


@pytest.mark.parametrize("text,expected", [
    ("we need it today", date(2026, 11, 2)),
    ("call me tomorrow", date(2026, 11, 3)),
    ("want to talk this week", date(2026, 11, 6)),
    ("by next week please", date(2026, 11, 13)),
    ("coverage ends at the end of the month", date(2026, 11, 30)),
    ("decision by the end of next month", date(2026, 12, 31)),
    ("expires in 3 weeks", date(2026, 11, 23)),
    ("expires in two weeks", date(2026, 11, 16)),
    ("expires in a couple of weeks", date(2026, 11, 16)),
    ("renews in 10 days", date(2026, 11, 12)),
    ("starting in 2 months", date(2027, 1, 2)),
    ("need it by Friday", date(2026, 11, 6)),
    ("call me on Monday", date(2026, 11, 2)),                   # today is Monday: on-or-after
    ("let's talk next Tuesday", date(2026, 11, 10)),
    ("policy ends Dec 5", date(2026, 12, 5)),
    ("effective December 5th, 2027", date(2027, 12, 5)),
    ("it expires 12/15", date(2026, 12, 15)),
    ("expires on 1/5/27", date(2027, 1, 5)),
    ("effective 5 December", date(2026, 12, 5)),
    ("it lapsed Oct 20", date(2026, 10, 20)),                   # recent past stays in the past
    ("start Jan 5", date(2027, 1, 5)),                          # clearly next year
])
def test_phrases_resolve_to_expected_dates(text, expected):
    assert d(text) == expected


@pytest.mark.parametrize("text", [
    "we run 3/4 ton pickups",
    "dispatch is 24/7",
    "we have 12 trucks and 14 drivers",
    "I called you on Tuesday about this",              # a past reference, not a deadline
    "I emailed last Friday",
    "we have been hauling since 2015",
    "",
])
def test_non_deadlines_are_ignored(text):
    assert d(text) is None


def test_earliest_date_wins():
    assert d("policy expires Dec 5 but please call me tomorrow") == date(2026, 11, 3)


def test_future_cue_overrides_past_cue():
    assert d("I sent an application and our policy expires on Tuesday") == date(2026, 11, 3)


def test_overlapping_phrases_keep_the_longer_one():
    matches = find_dates("by the end of next month", RECEIVED)
    assert len(matches) == 1 and matches[0].text.lower() == "end of next month"


def test_may_as_a_verb_is_not_a_month():
    assert d("we may need 5 trucks") is None


def test_resolve_from_extracted_texts_ignores_not_provided():
    # vague phrase: the SCORING date is the earliest plausible day, not the month end
    assert resolve_deadline_date(["not_provided", "Our policy expires next month"], RECEIVED) == date(2026, 12, 1)
    assert resolve_deadline_date(["not_provided", ""], RECEIVED) is None


def test_clause_around_is_a_verbatim_slice():
    text = "Hi there. Our insurance expires next month, so we need a quote. Thanks."
    hit = find_deadline(text, RECEIVED)
    clause = clause_around(text, hit)
    assert clause in text and "expires next month" in clause and "Hi there" not in clause
