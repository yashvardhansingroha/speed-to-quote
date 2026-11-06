"""Routing to human review vs scoring."""
from datetime import timedelta

from src.extraction.rules import extract_with_rules
from src.scoring.triage import triage_inquiry

SPEC = ("Hi, we run a trucking business with 12 trucks operating in Texas and Oklahoma. "
        "Our insurance expires next month. We're looking for a new quote and want to talk to someone this week.")


def run(text, checklist, now, **kw):
    ex = extract_with_rules(text, checklist, now)
    return triage_inquiry(text, ex, now, now, **kw), ex


def test_spec_example_is_scored_with_explanations(checklist, now):
    t, ex = run(SPEC, checklist, now)
    assert not t.needs_review and t.priority is not None
    assert t.days_until_deadline == 4                         # 'this week' -> Friday 6 Nov, received Mon 2 Nov
    assert t.priority.score == 25 + 40 + 0 + 0               # high urgency + <=7 day deadline, just received
    assert len(t.priority.explain()) == 4 and all(r.startswith("+") for r in t.priority.explain())


def test_unclear_intent_goes_to_human_review_without_a_score(checklist, now):
    t, ex = run("hello there my friend, how are things going", checklist, now)
    assert ex.intent == "unclear"
    assert t.needs_review and t.priority is None
    assert any("intent" in r.lower() for r in t.review_reasons)


def test_empty_and_very_short_messages_go_to_human_review(checklist, now):
    for text in ["", "   ", "Hi", "need quote asap"]:
        t, _ = run(text, checklist, now)
        assert t.needs_review and t.priority is None
        assert any("short" in r.lower() for r in t.review_reasons)


def test_waiting_hours_only_count_while_unanswered(checklist, now):
    received = now - timedelta(hours=24)
    ex = extract_with_rules("We need a quote for 3 trucks in Texas, no rush", checklist, received)
    waiting = triage_inquiry("We need a quote for 3 trucks in Texas, no rush", ex, received, now)
    answered = triage_inquiry("We need a quote for 3 trucks in Texas, no rush", ex, received, now,
                              first_response_at=received + timedelta(hours=2))
    assert waiting.hours_waiting == 24 and waiting.priority.factors[2].points == 10
    assert answered.hours_waiting is None and answered.priority.factors[2].points == 0


def test_overdue_followup_raises_the_score(checklist, now):
    base, _ = run(SPEC, checklist, now)
    overdue, _ = run(SPEC, checklist, now, followup="overdue")
    assert overdue.priority.score - base.priority.score == 15


def test_unverified_deadline_is_not_used(checklist, now):
    """If the deadline quote failed verification the status is 'unclear' and no date is derived."""
    ex = extract_with_rules(SPEC, checklist, now)
    ex.deadline.status, ex.deadline.value = "unclear", None
    t = triage_inquiry(SPEC, ex, now, now)
    assert t.deadline_date is None and t.priority.factors[1].points == 0
