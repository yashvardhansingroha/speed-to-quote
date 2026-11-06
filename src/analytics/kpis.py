"""Operations KPIs, computed from event timestamps in SQLite (nothing is hard-coded).

SYNTHETIC / DEMONSTRATION DATA. All "now" values are the fixed demo clock (config/settings.yaml).

Definitions (shown in the UI next to each number):
  Period            inquiries whose `received_at` DATE falls in [start, end] (both inclusive).
  First response    first_response event time minus received_at. Only inquiries that HAVE a first_response
                    event enter the median / P90. The rest are reported separately in two groups:
                      awaiting            status is still `open` and there is no first response (someone should act);
                      closed w/o response no first response but the inquiry is no longer open (e.g. closed as stale).
                    P90 is the 90th percentile with linear interpolation (pandas default).
  Overdue follow-up a follow-up task that is still `open` and whose due_at is before `now`.
                    Denominator = all follow-up tasks of inquiries in the period except `cancelled` ones.
  Completeness      provided fields / applicable fields (not_applicable removed), averaged over inquiries
                    that have at least one applicable field.
  Quote-to-bound    MATURE cohort only: quote-request inquiries (intent new_quote or renewal_or_replacement)
                    received at least `mature_cohort_days` before `now`, so most outcomes have had time to
                    happen. Rate = cohort inquiries with a `bound` event / cohort size.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

QUOTE_INTENTS = ("new_quote", "renewal_or_replacement")
INTENT_ORDER = ["new_quote", "renewal_or_replacement", "existing_policy_service", "follow_up",
                "document_or_info_provided", "coverage_question", "unclear"]


@dataclass(frozen=True)
class Ratio:
    numerator: int
    denominator: int

    @property
    def value(self) -> Optional[float]:
        """None (not 0) when the denominator is zero: 'no data' is different from '0%'."""
        return self.numerator / self.denominator if self.denominator else None

    @property
    def pct_text(self) -> str:
        return "n/a" if self.value is None else f"{self.value * 100:.1f}%"

    @property
    def fraction_text(self) -> str:
        return f"{self.numerator} of {self.denominator}"


@dataclass
class Frames:
    inquiries: pd.DataFrame      # id, received_at (datetime), status, channel
    extractions: pd.DataFrame    # inquiry_id, intent, urgency, needs_review
    events: pd.DataFrame         # inquiry_id, event_type, occurred_at (datetime)
    tasks: pd.DataFrame          # id, inquiry_id, task_type, due_at, status
    completeness: pd.DataFrame   # inquiry_id, provided, applicable


@dataclass
class MatureCohort:
    cutoff: date                 # inquiries received on/before this date are "mature"
    mature_days: int
    cohort_size: int
    bound: Ratio
    quoted: Ratio                # context: share of the cohort that reached the quote stage
    definition: str


@dataclass
class KPIs:
    start: date
    end: date
    total: int
    responded: int
    awaiting: int                # OPEN inquiries with no first response yet
    closed_without_response: int # closed (not open) inquiries that never got a first response
    response_hours: pd.Series    # one value per responded inquiry (kept for the histogram)
    median_hours: Optional[float]
    p90_hours: Optional[float]
    overdue: Ratio
    overdue_tasks: pd.DataFrame
    completeness_avg: Optional[float]
    completeness_n: int
    by_intent: pd.Series
    mature: MatureCohort
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- loading
def _ts(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c])
    return df


def load_frames(conn: sqlite3.Connection) -> Frames:
    inquiries = _ts(pd.read_sql_query("SELECT id, received_at, status, channel FROM inquiries", conn), "received_at")
    extractions = pd.read_sql_query(
        "SELECT inquiry_id, intent, urgency, needs_review FROM extractions "
        "WHERE id IN (SELECT MAX(id) FROM extractions GROUP BY inquiry_id)", conn)
    events = _ts(pd.read_sql_query("SELECT inquiry_id, event_type, occurred_at FROM events", conn), "occurred_at")
    tasks = _ts(pd.read_sql_query("SELECT id, inquiry_id, task_type, due_at, status FROM followup_tasks", conn), "due_at")
    completeness = pd.read_sql_query(
        "SELECT inquiry_id, SUM(status = 'provided') AS provided, SUM(status != 'not_applicable') AS applicable "
        "FROM checklist_status GROUP BY inquiry_id", conn)
    return Frames(inquiries, extractions, events, tasks, completeness)


def date_bounds(frames: Frames) -> tuple[date, date]:
    d = frames.inquiries["received_at"].dt.date
    return d.min(), d.max()


def filter_period(frames: Frames, start: date, end: date) -> Frames:
    """Keep only inquiries received in [start, end] and everything that belongs to them."""
    day = frames.inquiries["received_at"].dt.date
    inq = frames.inquiries[(day >= start) & (day <= end)]
    ids = set(inq["id"])
    return Frames(
        inq, frames.extractions[frames.extractions["inquiry_id"].isin(ids)],
        frames.events[frames.events["inquiry_id"].isin(ids)],
        frames.tasks[frames.tasks["inquiry_id"].isin(ids)],
        frames.completeness[frames.completeness["inquiry_id"].isin(ids)])


# --------------------------------------------------------------------------- individual metrics
def first_response_hours(frames: Frames) -> pd.Series:
    """Hours from receipt to first response, one value per inquiry that has been answered."""
    first = (frames.events[frames.events["event_type"] == "first_response"]
             .groupby("inquiry_id")["occurred_at"].min())
    recv = frames.inquiries.set_index("id")["received_at"]
    joined = pd.concat([recv, first.rename("first")], axis=1, join="inner")
    return ((joined["first"] - joined["received_at"]).dt.total_seconds() / 3600).rename("hours")


def overdue_followups(frames: Frames, now: datetime) -> tuple[Ratio, pd.DataFrame]:
    t = frames.tasks[frames.tasks["status"] != "cancelled"]
    over = t[(t["status"] == "open") & (t["due_at"] < pd.Timestamp(now))].copy()
    over["days_overdue"] = (pd.Timestamp(now) - over["due_at"]).dt.total_seconds() / 86400
    return Ratio(len(over), len(t)), over.sort_values("days_overdue", ascending=False)


def completeness_by_inquiry(frames: Frames) -> pd.Series:
    c = frames.completeness[frames.completeness["applicable"] > 0]
    return (c["provided"] / c["applicable"]).set_axis(c["inquiry_id"]).rename("completeness")


def intent_volume(frames: Frames) -> pd.Series:
    counts = frames.extractions["intent"].value_counts()
    return counts.reindex([i for i in INTENT_ORDER if i in counts.index]).astype(int)


def mature_cohort(frames: Frames, now: datetime, mature_days: int) -> MatureCohort:
    cutoff = (now - timedelta(days=mature_days)).date()
    ex = frames.extractions[frames.extractions["intent"].isin(QUOTE_INTENTS)]
    recv = frames.inquiries.set_index("id")["received_at"].dt.date
    cohort_ids = [i for i in ex["inquiry_id"] if recv.get(i) is not None and recv[i] <= cutoff]
    ev = frames.events[frames.events["inquiry_id"].isin(cohort_ids)]
    bound = ev[ev["event_type"] == "bound"]["inquiry_id"].nunique()
    quoted = ev[ev["event_type"] == "quote_sent"]["inquiry_id"].nunique()
    n = len(cohort_ids)
    return MatureCohort(
        cutoff=cutoff, mature_days=mature_days, cohort_size=n, bound=Ratio(bound, n), quoted=Ratio(quoted, n),
        definition=(f"Quote-request inquiries (intent new_quote or renewal_or_replacement) in the selected period that were "
                    f"received on or before {cutoff:%d %b %Y} (at least {mature_days} days before the demo date), so that "
                    "outcomes have had time to occur. Bound = the inquiry has a 'bound' event."))


# --------------------------------------------------------------------------- everything together
def compute_kpis(frames: Frames, now: datetime, mature_days: int, start: Optional[date] = None,
                 end: Optional[date] = None) -> KPIs:
    """Compute all six KPIs for the (already period-filtered) frames."""
    hours = first_response_hours(frames)
    total = len(frames.inquiries)
    responded = len(hours)
    comp = completeness_by_inquiry(frames)
    overdue, overdue_tasks = overdue_followups(frames, now)
    lo, hi = (date_bounds(frames) if total else (start, end))
    unanswered = frames.inquiries[~frames.inquiries["id"].isin(hours.index)]
    awaiting = int((unanswered["status"] == "open").sum())
    closed_no_response = len(unanswered) - awaiting
    notes = []
    if responded < total:
        notes.append(f"{total - responded} inquiries have no first response and are excluded from the response-time "
                     f"statistics: {awaiting} still open and awaiting one, {closed_no_response} closed without one.")
    return KPIs(
        start=start or lo, end=end or hi, total=total, responded=responded, awaiting=awaiting, closed_without_response=closed_no_response,
        response_hours=hours,
        median_hours=float(hours.median()) if responded else None,
        p90_hours=float(hours.quantile(0.9)) if responded else None,
        overdue=overdue, overdue_tasks=overdue_tasks,
        completeness_avg=float(comp.mean()) if len(comp) else None, completeness_n=len(comp),
        by_intent=intent_volume(frames), mature=mature_cohort(frames, now, mature_days), notes=notes)
