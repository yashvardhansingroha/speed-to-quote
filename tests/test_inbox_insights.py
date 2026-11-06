"""Priority inbox ranking and the rule-based insights (no causal claims, every number computed)."""
import re
from datetime import datetime

import pytest

from src.analysis import load_analysis
from src.analytics import kpis as K
from src.analytics.inbox import build_inbox
from src.analytics.insights import NUMBER_RX, build_insights

from tests.test_kpis import NOW, T, frames, responded


# ---- priority inbox ----------------------------------------------------------------------------------------
def test_inbox_is_sorted_by_score_then_inquiry_id(built_db, now):
    ib = build_inbox(built_db, now)
    r = ib.ranked
    assert list(r["Rank"]) == list(range(1, len(r) + 1))
    keys = list(zip(-r["Score"], r["Inquiry"]))
    assert keys == sorted(keys)


def test_inbox_contains_exactly_the_open_inquiries_split_into_ranked_and_review(built_db, now):
    ib = build_inbox(built_db, now)
    open_ids = {r[0] for r in built_db.execute("SELECT id FROM inquiries WHERE status='open'")}
    ranked, review = set(ib.ranked["Inquiry"]), set(ib.review["Inquiry"])
    assert ranked | review == open_ids and not (ranked & review)
    assert len(ranked) > 20 and len(review) > 0


def test_ranked_scores_match_the_triage_score_and_review_rows_have_no_score(built_db, now):
    ib = build_inbox(built_db, now)
    for _, row in ib.ranked.head(15).iterrows():
        a = load_analysis(built_db, int(row["Inquiry"]), now)
        assert not a.triage.needs_review and a.triage.priority.score == row["Score"]
    assert "Score" not in ib.review.columns and ib.review["Why"].str.len().min() > 0


def test_inbox_rows_carry_completeness_and_follow_up_status(built_db, now):
    ib = build_inbox(built_db, now)
    assert ib.ranked["Completeness"].str.fullmatch(r"\d+ of \d+").all()
    assert set(ib.ranked["Follow-up"]) <= {"overdue", "due", "none"}


def test_closed_bound_and_lost_inquiries_are_not_in_the_inbox(built_db, now):
    ib = build_inbox(built_db, now)
    done = {r[0] for r in built_db.execute("SELECT id FROM inquiries WHERE status != 'open'")}
    assert not (done & (set(ib.ranked["Inquiry"]) | set(ib.review["Inquiry"])))


# ---- insights --------------------------------------------------------------------------------------------------------
CAUSAL = re.compile(r"\b(because|caused?|causing|due to|led to|leads to|results? in|resulted|drives?|driven by|"
                    r"explains?|proves?|therefore)\b", re.IGNORECASE)
MONEY = re.compile(r"[$£€]|\b(revenue|profit|dollars?|income|worth|roi|savings?)\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def real_insights(built_db, now):
    f = K.load_frames(built_db)
    lo, hi = K.date_bounds(f)
    sub = K.filter_period(f, lo, hi)
    k = K.compute_kpis(sub, now, 45, lo, hi)
    return k, build_insights(sub, k, now)


def test_there_are_four_complete_insights(real_insights):
    _, ins = real_insights
    assert len(ins) == 4
    for i in ins:
        assert i.title and i.observation and i.action and i.metric and len(i.hypotheses) >= 2


def test_explanations_are_labelled_as_hypotheses(real_insights):
    for i in real_insights[1]:
        assert all(h.startswith("Hypothesis:") for h in i.hypotheses)


def test_every_number_in_an_observation_came_through_the_metric_formatter(real_insights):
    for i in real_insights[1]:
        used = set(NUMBER_RX.findall(i.observation))
        assert used and used <= i.printed_numbers, (i.title, used - i.printed_numbers)


def test_only_the_observation_contains_numbers(real_insights):
    for i in real_insights[1]:
        assert not NUMBER_RX.search(" ".join([*i.hypotheses, i.action, i.metric, i.title]))


def test_no_causal_or_revenue_language(real_insights):
    for i in real_insights[1]:
        text = " ".join([i.title, i.observation, *i.hypotheses, i.action, i.metric])
        assert not CAUSAL.search(text), CAUSAL.search(text).group(0)
        assert not MONEY.search(text), MONEY.search(text).group(0)


def test_observations_quote_the_computed_kpis(real_insights):
    k, ins = real_insights
    by = {i.title: i.observation for i in ins}
    assert f"{k.overdue.numerator} of {k.overdue.denominator} ({k.overdue.pct_text})" in by["Overdue follow-ups"]
    assert f"out of {k.responded} answered inquiries" in by["Response speed by stated urgency"]
    assert f"{k.start:%d %b %Y}" in by["Overdue follow-ups"] and f"{k.end:%d %b %Y}" in by["Overdue follow-ups"]


def test_insight_numbers_are_exact_for_a_hand_built_case():
    inq, ev = responded([1, 1, 2, 30])
    f = frames(inq, ev, intents=[(1, "new_quote"), (2, "new_quote"), (3, "follow_up"), (4, "follow_up")],
               tasks=[(1, 1, "missing_info", NOW.replace(day=1), "open"), (2, 2, "quote_follow_up", NOW.replace(day=30), "open"),
                      (3, 3, "missing_info", NOW.replace(day=1), "done")])
    f.extractions["urgency"] = ["high", "high", "low", "low"]
    k = K.compute_kpis(f, NOW, 45)
    ins = {i.title: i for i in build_insights(f, k, NOW)}
    assert "1 of 3 (33.3%)" in ins["Overdue follow-ups"].observation
    assert "1.0 h for high-urgency inquiries (n = 2)" in ins["Response speed by stated urgency"].observation
    assert "16.0 h for low-urgency inquiries (n = 2)" in ins["Response speed by stated urgency"].observation


def test_insights_degrade_gracefully_with_no_data():
    f = frames([(1, T(3), "open")])
    k = K.compute_kpis(f, NOW, 45)
    ins = build_insights(f, k, NOW)
    assert len(ins) == 4
    assert any("Not enough" in i.observation for i in ins)
    assert any("No follow-up tasks" in i.observation for i in ins)
