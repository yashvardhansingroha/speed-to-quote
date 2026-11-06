"""Rule-based operational insights. SYNTHETIC / DEMONSTRATION DATA.

Each insight has four parts: an OBSERVATION (with the exact numerator, denominator and period), POSSIBLE
EXPLANATIONS (labelled hypotheses), a SUGGESTED ACTION and a METRIC TO MONITOR.

Guardrails:
  * No causal claims and no revenue / money estimates.
  * Every number in an observation is produced through `Facts`, which records it, so a test can prove that no
    figure was typed by hand. The other three parts contain no numbers at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd

from src.analytics.kpis import QUOTE_INTENTS, Frames, KPIs, Ratio, completeness_by_inquiry, first_response_hours

NUMBER_RX = re.compile(r"\d+(?:\.\d+)?")


class Facts:
    """Formats numbers for the observation text and remembers every figure it printed."""

    def __init__(self) -> None:
        self.printed: set[str] = set()

    def _rec(self, text: str) -> str:
        self.printed.update(NUMBER_RX.findall(text))
        return text

    def n(self, value: int) -> str:
        return self._rec(str(int(value)))

    def hours(self, value: float) -> str:
        return self._rec(f"{value:.1f} h")

    def pct(self, r: Ratio) -> str:
        return self._rec(r.pct_text)

    def ratio(self, r: Ratio) -> str:
        return self._rec(f"{r.numerator} of {r.denominator} ({r.pct_text})")

    def day(self, d: date) -> str:
        return self._rec(f"{d:%d %b %Y}")


@dataclass
class Insight:
    title: str
    observation: str
    hypotheses: list[str]
    action: str
    metric: str
    printed_numbers: set[str] = field(default_factory=set)


def _period(f: Facts, k: KPIs) -> str:
    return f"between {f.day(k.start)} and {f.day(k.end)}"


def _median(s: pd.Series) -> Optional[float]:
    return float(s.median()) if len(s) else None


# --------------------------------------------------------------------------- the four insights
def insight_speed_by_urgency(frames: Frames, k: KPIs) -> Insight:
    f = Facts()
    hours = first_response_hours(frames)
    urg = frames.extractions.set_index("inquiry_id")["urgency"]
    by = {u: hours[hours.index.map(lambda i, u=u: urg.get(i) == u)] for u in ("high", "medium", "low")}
    med = {u: _median(s) for u, s in by.items()}
    if k.responded == 0 or med["high"] is None or med["low"] is None:
        obs = f"Not enough answered inquiries received {_period(f, k)} to compare response speed by stated urgency."
    else:
        obs = (f"For inquiries received {_period(f, k)}, median first-response time was {f.hours(med['high'])} for high-urgency "
               f"inquiries (n = {f.n(len(by['high']))}) and {f.hours(med['low'])} for low-urgency inquiries "
               f"(n = {f.n(len(by['low']))}), out of {f.n(k.responded)} answered inquiries.")
    return Insight(
        "Response speed by stated urgency", obs,
        ["Hypothesis: staff may already pick up urgent-sounding messages first.",
         "Hypothesis: urgent messages may tend to arrive during staffed hours.",
         "Hypothesis: small groups make the medians noisy, so a gap may not persist."],
        "Make the priority inbox the first screen for whoever answers new inquiries, then re-check this comparison.",
        "Median first-response hours by stated urgency, tracked weekly.", f.printed)


def insight_awaiting(frames: Frames, k: KPIs, now: datetime) -> Insight:
    f = Facts()
    ev = frames.events
    answered = set(ev[ev["event_type"] == "first_response"]["inquiry_id"])
    open_ids = frames.inquiries[frames.inquiries["status"] == "open"]
    waiting = open_ids[~open_ids["id"].isin(answered)]
    over_day = waiting[(pd.Timestamp(now) - waiting["received_at"]) > pd.Timedelta(hours=24)]
    r_wait = Ratio(len(waiting), len(open_ids))
    r_old = Ratio(len(over_day), len(waiting))
    if len(open_ids) == 0:
        obs = f"No open inquiries received {_period(f, k)}."
    else:
        obs = (f"Among {f.n(len(open_ids))} open inquiries received {_period(f, k)}, {f.ratio(r_wait)} have no first "
               f"response; {f.ratio(r_old)} of those have waited longer than {f.n(24)} h as of the demo date.")
    return Insight(
        "Open inquiries still waiting for a first response", obs,
        ["Hypothesis: messages received outside staffed hours or at weekends queue until the next working day.",
         "Hypothesis: some inquiries may be duplicates or low-intent messages that nobody owns.",
         "Hypothesis: capacity may be stretched on days with many arrivals."],
        "Assign an owner to every inquiry with no first response and clear the oldest first.",
        "Count of open inquiries with no first response older than one day, checked daily.", f.printed)


def insight_overdue(k: KPIs) -> Insight:
    f = Facts()
    o = k.overdue
    kinds = k.overdue_tasks["task_type"].value_counts()
    if o.denominator == 0:
        obs = f"No follow-up tasks for inquiries received {_period(f, k)}."
    else:
        parts = ", ".join(f"{f.n(v)} {name.replace('_', ' ')}" for name, v in kinds.items()) or "none"
        obs = (f"For inquiries received {_period(f, k)}, {f.ratio(o)} follow-up tasks are overdue (open and past due). "
               f"By type: {parts}.")
    return Insight(
        "Overdue follow-ups", obs,
        ["Hypothesis: follow-up reminders may live outside the shared workflow.",
         "Hypothesis: drafts may be waiting for a human to approve them.",
         "Hypothesis: tasks may stay open after the customer already replied by phone."],
        "Review the overdue list daily and approve or reject waiting drafts before starting new work.",
        "Overdue follow-ups as a share of all non-cancelled follow-up tasks.", f.printed)


def insight_completeness_vs_quote(frames: Frames, k: KPIs) -> Insight:
    f = Facts()
    comp = completeness_by_inquiry(frames)
    ex = frames.extractions[frames.extractions["intent"].isin(QUOTE_INTENTS)]
    ids = [i for i in ex["inquiry_id"] if i in comp.index]
    quoted = set(frames.events[frames.events["event_type"] == "quote_sent"]["inquiry_id"])
    if len(ids) < 4:
        obs = f"Not enough quote-request inquiries received {_period(f, k)} to compare completeness bands."
    else:
        cut = float(comp[ids].median())
        hi = [i for i in ids if comp[i] >= cut]
        lo = [i for i in ids if comp[i] < cut]
        r_hi = Ratio(sum(i in quoted for i in hi), len(hi))
        r_lo = Ratio(sum(i in quoted for i in lo), len(lo))
        obs = (f"Among {f.n(len(ids))} quote-request inquiries received {_period(f, k)}, {f.ratio(r_hi)} of those at or "
               f"above the median completeness ({f.pct(Ratio(round(cut * 1000), 1000))}) reached the quote stage, "
               f"against {f.ratio(r_lo)} of those below it.")
    return Insight(
        "Completeness and reaching the quote stage", obs,
        ["Hypothesis: more complete submissions may be quicker to quote.",
         "Hypothesis: brokers may chase promising leads harder, so completeness may partly reflect engagement.",
         "Hypothesis: this dataset is synthetic, so any gap may simply reflect how it was generated."],
        "Ask for the missing items on quote requests in the first reply, then watch whether the gap narrows.",
        "Share of quote requests reaching the quote stage, split by completeness band.", f.printed)


def build_insights(frames: Frames, k: KPIs, now: datetime) -> list[Insight]:
    return [insight_speed_by_urgency(frames, k), insight_awaiting(frames, k, now), insight_overdue(k),
            insight_completeness_vs_quote(frames, k)]
