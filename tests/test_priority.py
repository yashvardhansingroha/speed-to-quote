"""Priority formula: known inputs -> known score; weights come from config."""
import copy
from datetime import datetime, timedelta

from src.config import load_priority_weights
from src.scoring.priority import PriorityInputs, compute_priority, followup_status


def test_known_inputs_give_known_score():
    # urgency high 25 + deadline in 21 days (<=30 tier) 30 + waiting 24h of 48h = 10 + overdue follow-up 15
    r = compute_priority(PriorityInputs("high", 21, "expires next month", 24.0, "overdue"))
    assert r.score == 80
    assert [f.points for f in r.factors] == [25, 30, 10, 15]


def test_all_zero_inputs_score_zero():
    assert compute_priority(PriorityInputs("low")).score == 0


def test_maximum_possible_score_is_100():
    r = compute_priority(PriorityInputs("high", 0, "today", 500.0, "overdue"))
    assert r.score == 100


def test_each_factor_has_plain_english_reason():
    r = compute_priority(PriorityInputs("medium", 21, "next month", None, "due"))
    reasons = r.explain()
    assert reasons[0].startswith("+12:") and "medium" in reasons[0]
    assert reasons[1].startswith("+30:") and "21 days" in reasons[1]
    assert "first response" in reasons[2]                 # already answered -> 0 waiting points
    assert reasons[3].startswith("+8:")


def test_deadline_tiers():
    pts = lambda d: compute_priority(PriorityInputs("low", d)).factors[1].points
    assert [pts(3), pts(7), pts(8), pts(14), pts(15), pts(30), pts(45), pts(75), pts(200)] == [40, 40, 35, 35, 30, 30, 18, 8, 0]
    assert pts(None) == 0


def test_deadline_in_the_past_gets_top_tier_and_says_so():
    f = compute_priority(PriorityInputs("low", -3, "policy lapsed")).factors[1]
    assert f.points == 40 and "passed 3 days ago" in f.reason


def test_waiting_ramps_linearly_and_caps():
    wait = lambda h: compute_priority(PriorityInputs("low", None, None, h)).factors[2].points
    assert [wait(0), wait(12), wait(24), wait(48), wait(500)] == [0, 5, 10, 20, 20]


def test_weights_are_read_from_config_not_hardcoded():
    w = copy.deepcopy(load_priority_weights())
    w["urgency"]["points"]["high"] = 10                  # change the config values ...
    w["urgency"]["max_points"] = 10
    base = compute_priority(PriorityInputs("high")).score
    changed = compute_priority(PriorityInputs("high"), w).score
    assert base == 25 and changed == 10                  # ... and the score follows


def test_default_weights_match_yaml_file():
    w = load_priority_weights()
    assert compute_priority(PriorityInputs("high")).score == w["urgency"]["points"]["high"]


def test_followup_status_classification():
    now = datetime(2026, 11, 2, 9, 0)
    assert followup_status([], now) == "none"
    assert followup_status([now - timedelta(hours=1)], now) == "overdue"
    assert followup_status([now + timedelta(hours=5)], now) == "due"
    assert followup_status([now + timedelta(hours=72)], now) == "none"
    assert followup_status([now + timedelta(hours=72), now - timedelta(days=2)], now) == "overdue"


def test_a_deadline_lapsed_long_ago_is_treated_as_stale_not_urgent():
    recent = compute_priority(PriorityInputs("low", -30, "Oct 3")).factors[1]
    old = compute_priority(PriorityInputs("low", -31, "Jul 3")).factors[1]
    assert recent.points == 40 and old.points == 0
    assert "likely stale" in old.reason and "31 days ago" in old.reason


def test_the_lapsed_window_comes_from_config():
    w = copy.deepcopy(load_priority_weights())
    w["deadline"]["lapsed_window_days"] = 5
    assert compute_priority(PriorityInputs("low", -6, "x"), w).factors[1].points == 0
    assert compute_priority(PriorityInputs("low", -5, "x"), w).factors[1].points == 40
