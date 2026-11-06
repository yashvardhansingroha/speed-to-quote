"""Simulate a consistent event history for each SYNTHETIC inquiry.

    inquiry_created -> first_response -> (followup_sent / customer_reply)* -> quote_sent -> bound | lost
                                                                  or -> closed  (non-quote inquiries)

Guarantees (checked by tests/test_database.py):
  * events are in time order, none after the reference "now";
  * no response before creation; no outcome (bound/lost) without an earlier quote_sent;
  * at most one terminal event, and nothing after it;
  * follow-up tasks that are still open belong only to open inquiries.

ASSUMPTIONS BAKED INTO THE SYNTHETIC DATA (so nobody mistakes them for findings):
  * response delay is log-normal, a bit faster for high-urgency inquiries, and pushed to the next
    business morning when it lands outside weekday 08:00-18:00;
  * ~4% of inquiries never get a response;
  * quote probability rises modestly with the number of checklist fields supplied;
  * bound probability is modestly higher when the first response was fast.
These make the dashboard show something, but they are inputs to the simulation, not evidence.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

QUOTE_INTENTS = {"new_quote", "renewal_or_replacement"}


@dataclass
class Event:
    event_type: str
    occurred_at: datetime
    detail: Optional[str] = None


@dataclass
class Task:
    task_type: str
    created_at: datetime
    due_at: datetime
    status: str = "open"                    # open | done | cancelled
    completed_at: Optional[datetime] = None


@dataclass
class History:
    events: list[Event] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    status: str = "open"                    # open | closed | bound | lost


def _business_hours(dt: datetime, rng: random.Random) -> datetime:
    """Push a timestamp that falls at night / weekend to the next weekday morning."""
    if dt.weekday() < 5 and 8 <= dt.hour < 18:
        return dt
    nxt = dt if dt.hour < 8 else dt + timedelta(days=1)
    nxt = nxt.replace(hour=8, minute=0, second=0, microsecond=0)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt + timedelta(minutes=rng.randint(0, 90))


def _lognormal_hours(rng: random.Random, median: float, sigma: float) -> float:
    return rng.lognormvariate(math.log(median), sigma)


def simulate_history(rng: random.Random, received_at: datetime, intent: str, urgency: str,
                     n_provided: int, now: datetime) -> History:
    h = History(events=[Event("inquiry_created", received_at)])

    def add(kind: str, when: datetime, detail: Optional[str] = None) -> bool:
        """Add an event only if it has already happened by `now`."""
        if when > now:
            return False
        h.events.append(Event(kind, when, detail))
        return True

    # ---- first response -------------------------------------------------------------
    median = {"high": 2.0, "medium": 4.0, "low": 8.0}[urgency]
    resp_at = _business_hours(received_at + timedelta(hours=max(0.1, _lognormal_hours(rng, median, 1.0))), rng)
    if rng.random() < 0.05:                                    # slow tail: 1-4 days
        resp_at += timedelta(hours=rng.uniform(24, 96))
    never_responded = rng.random() < 0.04

    if never_responded:
        # Stale: if it has been sitting 14+ days we administratively close it.
        stale_at = received_at + timedelta(days=14)
        if add("closed", stale_at, "no response / stale"):
            h.status = "closed"
        return h
    if not add("first_response", resp_at):
        return h                                              # still waiting for a first response

    clock = resp_at
    quote_intent = intent in QUOTE_INTENTS

    # ---- missing-info follow-up ------------------------------------------------------
    n_missing = 12 - n_provided
    if intent not in ("document_or_info_provided",) and (n_missing >= 4 or rng.random() < 0.3):
        due = resp_at + timedelta(hours=rng.uniform(24, 72))
        task = Task("missing_info", resp_at, due)
        h.tasks.append(task)
        if rng.random() < 0.78:                               # a follow-up actually gets sent
            late = rng.random() < 0.25
            sent_at = due + timedelta(hours=rng.uniform(24, 96) if late else rng.uniform(-20, 8))
            sent_at = max(sent_at, resp_at + timedelta(hours=1))
            if add("followup_sent", sent_at):
                task.status, task.completed_at = "done", sent_at
                clock = sent_at
                if rng.random() < 0.65:                       # customer answers
                    reply_at = sent_at + timedelta(hours=_lognormal_hours(rng, 20, 0.9))
                    if add("customer_reply", reply_at, "provided requested info"):
                        clock = reply_at

    # ---- quote and outcome (only quote requests) --------------------------------------
    if quote_intent and not any(e.event_type == "closed" for e in h.events):
        p_quote = min(0.8, max(0.2, 0.30 + 0.04 * n_provided))
        if rng.random() < p_quote:
            quote_at = clock + timedelta(days=rng.uniform(1, 7))
            if add("quote_sent", quote_at):
                fast = (resp_at - received_at) <= timedelta(hours=4)
                p_bound = 0.34 if fast else 0.22
                r = rng.random()
                outcome = "bound" if r < p_bound else ("lost" if r < p_bound + 0.40 else None)
                follow = Task("quote_follow_up", quote_at, quote_at + timedelta(days=3))
                h.tasks.append(follow)
                if outcome:
                    out_at = quote_at + timedelta(days=rng.uniform(2, 21))
                    if add(outcome, out_at):
                        h.status = outcome
                if h.status == "open" and rng.random() < 0.6:     # nudge sent on time-ish
                    nudge = follow.due_at + timedelta(hours=rng.uniform(-6, 30))
                    if add("followup_sent", nudge):
                        follow.status, follow.completed_at = "done", nudge
    elif not quote_intent and rng.random() < 0.75:
        close_at = clock + timedelta(days=rng.uniform(0.1, 4))     # always after the last event
        if add("closed", close_at, "resolved"):
            h.status = "closed"

    # ---- terminal clean-up -----------------------------------------------------------
    if h.status != "open":
        end = max(e.occurred_at for e in h.events)
        for t in h.tasks:
            if t.status == "open":
                t.status, t.completed_at = "cancelled", end
    h.events.sort(key=lambda e: e.occurred_at)
    return h
