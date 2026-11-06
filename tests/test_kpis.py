"""KPI calculations: medians, P90, zero denominators, open inquiries with no response, period filter, mature cohorts."""
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.analytics import kpis as K

NOW = datetime(2026, 11, 2, 9, 0)
T = lambda days_ago, hour=9: NOW - timedelta(days=days_ago) + timedelta(hours=hour - 9)


def frames(inquiries=(), events=(), tasks=(), intents=(), completeness=()):
    """Hand-built frames. inquiries: (id, received_at, status); events: (id, type, when);
    tasks: (task_id, inquiry_id, type, due_at, status); intents: (inquiry_id, intent); completeness: (id, provided, applicable)."""
    inq = pd.DataFrame(list(inquiries), columns=["id", "received_at", "status"])
    inq["channel"] = "email"
    inq["received_at"] = pd.to_datetime(inq["received_at"])
    ev = pd.DataFrame(list(events), columns=["inquiry_id", "event_type", "occurred_at"])
    ev["occurred_at"] = pd.to_datetime(ev["occurred_at"])
    tk = pd.DataFrame(list(tasks), columns=["id", "inquiry_id", "task_type", "due_at", "status"])
    tk["due_at"] = pd.to_datetime(tk["due_at"])
    ex = pd.DataFrame(list(intents), columns=["inquiry_id", "intent"])
    ex["urgency"], ex["needs_review"] = "medium", 0
    cp = pd.DataFrame(list(completeness), columns=["inquiry_id", "provided", "applicable"])
    return K.Frames(inq, ex, ev, tk, cp)


def responded(hours_list, start_id=1):
    """Inquiries answered after the given number of hours."""
    inq, ev = [], []
    for n, h in enumerate(hours_list, start_id):
        rec = T(10)
        inq.append((n, rec, "open"))
        ev.append((n, "inquiry_created", rec))
        ev.append((n, "first_response", rec + timedelta(hours=h)))
    return inq, ev


# ---- first response: median, P90, awaiting ---------------------------------------------------------
def test_median_and_p90_of_first_response_time():
    inq, ev = responded([1, 2, 3, 4, 10])
    k = K.compute_kpis(frames(inq, ev), NOW, 45)
    assert k.median_hours == 3.0
    assert k.p90_hours == pytest.approx(7.6)               # linear interpolation: 4 + 0.6 * (10 - 4)
    assert (k.total, k.responded, k.awaiting) == (5, 5, 0)


def test_even_count_median_is_the_mean_of_the_middle_two():
    inq, ev = responded([1, 2, 3, 10])
    assert K.compute_kpis(frames(inq, ev), NOW, 45).median_hours == 2.5


def test_open_inquiries_with_no_response_are_excluded_from_timing_and_counted_separately():
    inq, ev = responded([2, 4])
    inq += [(3, T(5), "open"), (4, T(1), "open")]                      # two inquiries nobody has answered
    ev += [(3, "inquiry_created", T(5)), (4, "inquiry_created", T(1))]
    k = K.compute_kpis(frames(inq, ev), NOW, 45)
    assert (k.total, k.responded, k.awaiting) == (4, 2, 2)
    assert k.median_hours == 3.0                                       # not dragged up by the unanswered ones
    assert any("2 inquiries have no first response" in n for n in k.notes)


def test_nobody_answered_gives_none_not_zero():
    inq = [(1, T(3), "open")]
    k = K.compute_kpis(frames(inq, [(1, "inquiry_created", T(3))]), NOW, 45)
    assert k.median_hours is None and k.p90_hours is None and k.responded == 0 and k.awaiting == 1


def test_only_the_first_response_counts():
    inq, ev = responded([5])
    ev.append((1, "first_response", T(10) + timedelta(hours=50)))     # a duplicate later event is ignored
    assert K.compute_kpis(frames(inq, ev), NOW, 45).median_hours == 5.0


# ---- overdue follow-ups -----------------------------------------------------------------------------
def tasks_frame():
    inq = [(1, T(20), "open"), (2, T(20), "open"), (3, T(20), "closed")]
    tasks = [(1, 1, "missing_info", NOW - timedelta(days=2), "open"),       # overdue
             (2, 1, "quote_follow_up", NOW + timedelta(days=2), "open"),    # open but not yet due
             (3, 2, "missing_info", NOW - timedelta(days=5), "done"),       # done: not overdue, still in denominator
             (4, 3, "missing_info", NOW - timedelta(days=9), "cancelled"),  # cancelled: excluded entirely
             (5, 2, "quote_follow_up", NOW - timedelta(hours=1), "open")]   # overdue
    return frames(inq, [], tasks)


def test_overdue_numerator_and_denominator():
    k = K.compute_kpis(tasks_frame(), NOW, 45)
    assert (k.overdue.numerator, k.overdue.denominator) == (2, 4)
    assert k.overdue.value == 0.5 and k.overdue.pct_text == "50.0%" and k.overdue.fraction_text == "2 of 4"


def test_overdue_list_is_sorted_most_overdue_first():
    o = K.compute_kpis(tasks_frame(), NOW, 45).overdue_tasks
    assert list(o["id"]) == [1, 5] and o["days_overdue"].iloc[0] == pytest.approx(2.0)


def test_zero_followup_tasks_gives_na_not_a_division_error():
    k = K.compute_kpis(frames([(1, T(3), "open")]), NOW, 45)
    assert (k.overdue.numerator, k.overdue.denominator) == (0, 0)
    assert k.overdue.value is None and k.overdue.pct_text == "n/a"


def test_all_tasks_cancelled_is_a_zero_denominator():
    f = frames([(1, T(3), "closed")], tasks=[(1, 1, "missing_info", NOW - timedelta(days=1), "cancelled")])
    assert K.compute_kpis(f, NOW, 45).overdue.pct_text == "n/a"


# ---- completeness -----------------------------------------------------------------------------------
def test_average_completeness_uses_applicable_fields_only():
    f = frames([(1, T(3), "open"), (2, T(3), "open"), (3, T(3), "open")],
               completeness=[(1, 3, 10), (2, 5, 5), (3, 0, 0)])        # third has nothing applicable: excluded
    k = K.compute_kpis(f, NOW, 45)
    assert k.completeness_avg == pytest.approx((0.3 + 1.0) / 2) and k.completeness_n == 2


def test_completeness_with_no_data_is_none():
    assert K.compute_kpis(frames([(1, T(3), "open")]), NOW, 45).completeness_avg is None


# ---- intent volume ------------------------------------------------------------------------------------
def test_intent_volume_counts_and_order():
    f = frames([(i, T(3), "open") for i in range(1, 7)],
               intents=[(1, "follow_up"), (2, "new_quote"), (3, "new_quote"), (4, "unclear"), (5, "new_quote"), (6, "follow_up")])
    v = K.compute_kpis(f, NOW, 45).by_intent
    assert v.to_dict() == {"new_quote": 3, "follow_up": 2, "unclear": 1} and v.sum() == 6


# ---- mature cohort ---------------------------------------------------------------------------------------
def cohort_frames():
    inq = [(1, T(60), "bound"), (2, T(60), "lost"), (3, T(50), "open"), (4, T(10), "open"), (5, T(70), "closed"),
           (6, T(45), "bound")]
    ev = [(1, "quote_sent", T(55)), (1, "bound", T(50)), (2, "quote_sent", T(55)), (2, "lost", T(50)),
          (3, "quote_sent", T(45)), (4, "quote_sent", T(8)), (6, "bound", T(30)), (6, "quote_sent", T(40))]
    intents = [(1, "new_quote"), (2, "renewal_or_replacement"), (3, "new_quote"), (4, "new_quote"),
               (5, "existing_policy_service"), (6, "new_quote")]
    return frames(inq, ev, intents=intents)


def test_mature_cohort_counts_only_old_quote_requests():
    m = K.compute_kpis(cohort_frames(), NOW, 45).mature
    # #4 is too recent (10 days) and #5 is not a quote request; #6 is exactly 45 days old, so it is included.
    assert m.cohort_size == 4 and (m.bound.numerator, m.bound.denominator) == (2, 4)
    assert m.bound.value == 0.5 and m.cutoff == (NOW - timedelta(days=45)).date()
    assert (m.quoted.numerator, m.quoted.denominator) == (4, 4)


def test_changing_the_cutoff_changes_the_cohort():
    assert K.compute_kpis(cohort_frames(), NOW, 5).mature.cohort_size == 5
    assert K.compute_kpis(cohort_frames(), NOW, 55).mature.cohort_size == 2     # only the two received 60 days ago
    assert K.compute_kpis(cohort_frames(), NOW, 65).mature.cohort_size == 0     # the only 70-day-old one is not a quote request


def test_empty_mature_cohort_is_na_and_describes_itself():
    f = frames([(1, T(3), "open")], intents=[(1, "new_quote")])
    m = K.compute_kpis(f, NOW, 45).mature
    assert m.cohort_size == 0 and m.bound.value is None and m.bound.pct_text == "n/a"
    assert "45 days" in m.definition and "bound" in m.definition


# ---- period filter ---------------------------------------------------------------------------------------------
def test_period_filter_is_inclusive_and_carries_events_and_tasks_along():
    inq = [(1, datetime(2026, 10, 1, 8), "open"), (2, datetime(2026, 10, 15, 23, 59), "open"), (3, datetime(2026, 10, 16, 0, 1), "open")]
    ev = [(i, "inquiry_created", datetime(2026, 10, 1)) for i in (1, 2, 3)]
    tk = [(1, 1, "missing_info", NOW, "open"), (2, 3, "missing_info", NOW, "open")]
    f = K.filter_period(frames(inq, ev, tk), date(2026, 10, 1), date(2026, 10, 15))
    assert list(f.inquiries["id"]) == [1, 2]                           # first and last day both included
    assert set(f.events["inquiry_id"]) == {1, 2} and list(f.tasks["inquiry_id"]) == [1]


def test_kpis_follow_the_period():
    inq, ev = responded([1, 1, 1])
    inq = [(i, datetime(2026, 10, d), s) for (i, _, s), d in zip(inq, (1, 10, 20))]
    ev = [(i, t, datetime(2026, 10, {1: 1, 2: 10, 3: 20}[i]) + timedelta(hours=1 if t == "first_response" else 0)) for i, t, _ in ev]
    full = frames(inq, ev)
    assert K.compute_kpis(K.filter_period(full, date(2026, 10, 1), date(2026, 10, 31)), NOW, 45).total == 3
    assert K.compute_kpis(K.filter_period(full, date(2026, 10, 5), date(2026, 10, 15)), NOW, 45).total == 1


def test_an_empty_period_does_not_crash():
    f = K.filter_period(frames([(1, T(3), "open")]), date(2020, 1, 1), date(2020, 1, 2))
    k = K.compute_kpis(f, NOW, 45, date(2020, 1, 1), date(2020, 1, 2))
    assert k.total == 0 and k.median_hours is None and k.overdue.pct_text == "n/a" and k.by_intent.empty


# ---- cross-check against independent SQL on the real (synthetic) database ------------------------------------------
def test_kpis_match_independent_sql_calculations(built_db, now):
    k = K.compute_kpis(K.load_frames(built_db), now, 45)
    q = lambda sql: built_db.execute(sql).fetchone()[0]
    assert k.total == q("SELECT COUNT(*) FROM inquiries")
    assert k.responded == q("SELECT COUNT(DISTINCT inquiry_id) FROM events WHERE event_type='first_response'")
    open_no_response = q("SELECT COUNT(*) FROM inquiries i WHERE i.status = 'open' AND NOT EXISTS (SELECT 1 FROM events e "
                         "WHERE e.inquiry_id = i.id AND e.event_type = 'first_response')")
    assert k.awaiting == open_no_response                      # open AND unanswered only
    assert k.awaiting + k.closed_without_response == k.total - k.responded
    assert (k.overdue.numerator, k.overdue.denominator) == (
        q(f"SELECT COUNT(*) FROM followup_tasks WHERE status='open' AND due_at < '{now:%Y-%m-%dT%H:%M:%S}'"),
        q("SELECT COUNT(*) FROM followup_tasks WHERE status != 'cancelled'"))
    hours = sorted(r[0] for r in built_db.execute(
        "SELECT (julianday(e.occurred_at) - julianday(i.received_at)) * 24 FROM events e JOIN inquiries i ON i.id = e.inquiry_id "
        "WHERE e.event_type = 'first_response'"))
    mid = len(hours) // 2
    expected_median = hours[mid] if len(hours) % 2 else (hours[mid - 1] + hours[mid]) / 2
    assert k.median_hours == pytest.approx(expected_median, abs=1e-3)
    assert k.by_intent.sum() == k.total


def test_load_frames_uses_only_the_latest_extraction_per_inquiry(built_db, now):
    built_db.execute("INSERT INTO extractions (inquiry_id, method, intent, urgency, needs_review, raw_json, created_at) "
                     "VALUES (1, 'llm', 'coverage_question', 'low', 0, '{}', '2026-11-02T09:00:00')")
    built_db.commit()
    f = K.load_frames(built_db)
    assert len(f.extractions) == len(f.inquiries)
    assert f.extractions[f.extractions["inquiry_id"] == 1]["intent"].iloc[0] == "coverage_question"
    built_db.execute("DELETE FROM extractions WHERE method='llm'")
    built_db.commit()


def test_open_and_awaiting_is_distinguished_from_closed_without_a_response():
    inq, ev = responded([2])
    inq += [(2, T(5), "open"), (3, T(9), "closed"), (4, T(9), "closed"), (5, T(9), "lost")]
    ev += [(i, "inquiry_created", T(5)) for i in (2, 3, 4, 5)]
    k = K.compute_kpis(frames(inq, ev), NOW, 45)
    assert (k.total, k.responded) == (5, 1)
    assert (k.awaiting, k.closed_without_response) == (1, 3)
    assert k.awaiting + k.closed_without_response == k.total - k.responded
    assert any("1 still open and awaiting one, 3 closed without one" in n for n in k.notes)


def test_a_closed_unanswered_inquiry_is_not_awaiting():
    k = K.compute_kpis(frames([(1, T(30), "closed")], [(1, "inquiry_created", T(30))]), NOW, 45)
    assert k.awaiting == 0 and k.closed_without_response == 1


def test_the_awaiting_insight_and_the_kpi_agree(built_db, now):
    from src.analytics.insights import insight_awaiting
    f = K.load_frames(built_db)
    lo, hi = K.date_bounds(f)
    sub = K.filter_period(f, lo, hi)
    k = K.compute_kpis(sub, now, 45, lo, hi)
    ins = insight_awaiting(sub, k, now)
    assert f"{k.awaiting} of " in ins.observation and k.closed_without_response > 0
