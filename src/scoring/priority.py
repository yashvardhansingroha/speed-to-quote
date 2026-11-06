"""Explainable priority score (0-100). Pure function: same inputs -> same score, always.

    score = urgency points + deadline points + waiting points + follow-up points

Weights and tiers come from config/priority_weights.yaml (never hard-coded here).
This is OPERATIONAL urgency - "how soon should a human look at this" - and is not a prediction
of conversion, deal size or customer quality. No personal or protected attribute is an input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.config import load_priority_weights


@dataclass(frozen=True)
class PriorityInputs:
    urgency: str                                   # high | medium | low
    days_until_deadline: Optional[int] = None      # None = no stated deadline; negative = already passed
    deadline_label: Optional[str] = None           # the customer's words, for the explanation
    hours_waiting: Optional[float] = None          # hours since received, ONLY if still unanswered; else None
    followup_status: str = "none"                  # overdue | due | none
    deadline_approximate: bool = False             # True when the words give a range (e.g. "next month")


@dataclass(frozen=True)
class Factor:
    name: str
    points: int
    max_points: int
    reason: str                                    # plain-English explanation, e.g. "+30: ..."


@dataclass
class PriorityResult:
    score: int
    factors: list[Factor] = field(default_factory=list)

    def explain(self) -> list[str]:
        return [f.reason for f in self.factors]


def _round_half_up(x: float) -> int:
    return int(x + 0.5)


def _fmt(points: int, text: str) -> str:
    return f"+{points}: {text}"


def _urgency_factor(urgency: str, cfg: dict[str, Any]) -> Factor:
    pts = int(cfg["points"].get(urgency, 0))
    return Factor("urgency", pts, cfg["max_points"], _fmt(pts, f"customer's stated urgency is {urgency}"))


def _deadline_factor(days: Optional[int], label: Optional[str], cfg: dict[str, Any],
                     approximate: bool = False) -> Factor:
    mx = cfg["max_points"]
    if days is None:
        return Factor("deadline", 0, mx, _fmt(0, "no explicit deadline stated"))
    window = int(cfg.get("lapsed_window_days", 30))
    if days < -window:      # lapsed long ago: probably stale, so no urgency points
        quoted = f' ("{label}")' if label else ""
        return Factor("deadline", 0, mx, _fmt(0, f"deadline{quoted} passed {-days} days ago: likely stale, "
                                                 "confirm it is still relevant"))
    pts = int(cfg["beyond_tiers_points"])
    for tier in sorted(cfg["tiers"], key=lambda t: t["max_days"]):
        if days <= tier["max_days"]:                # a past deadline (negative days) lands in the first tier
            pts = int(tier["points"])
            break
    quoted = f' ("{label}")' if label else ""
    if days < 0:
        when = f"deadline{quoted} passed {-days} day{'s' if -days != 1 else ''} ago"
    elif days == 0:
        when = f"deadline{quoted} is today"
    else:
        when = f"deadline{quoted} is {days} day{'s' if days != 1 else ''} away"
    if approximate:
        when += " at the earliest (approximate; confirm the exact date)"
    return Factor("deadline", pts, mx, _fmt(pts, when))


def _waiting_factor(hours: Optional[float], cfg: dict[str, Any]) -> Factor:
    mx = cfg["max_points"]
    if hours is None:
        return Factor("waiting", 0, mx, _fmt(0, "already had a first response"))
    frac = min(max(hours, 0.0) / cfg["full_points_at_hours"], 1.0)
    pts = _round_half_up(mx * frac)
    return Factor("waiting", pts, mx, _fmt(pts, f"waiting {hours:.0f}h with no response"))


def _followup_factor(status: str, cfg: dict[str, Any]) -> Factor:
    pts = int(cfg["points"].get(status, 0))
    text = {"overdue": "a follow-up is overdue", "due": "a follow-up is due soon"}.get(
        status, "no follow-up pending")
    return Factor("followup", pts, cfg["max_points"], _fmt(pts, text))


def compute_priority(inp: PriorityInputs, weights: Optional[dict[str, Any]] = None) -> PriorityResult:
    """Score one inquiry and return the per-factor breakdown."""
    w = weights or load_priority_weights()
    factors = [
        _urgency_factor(inp.urgency, w["urgency"]),
        _deadline_factor(inp.days_until_deadline, inp.deadline_label, w["deadline"], inp.deadline_approximate),
        _waiting_factor(inp.hours_waiting, w["waiting"]),
        _followup_factor(inp.followup_status, w["followup"]),
    ]
    # Each factor is already capped by config validation; min() is a belt-and-braces guarantee.
    return PriorityResult(score=min(100, sum(f.points for f in factors)), factors=factors)


def followup_status(open_task_due_times: list[datetime], now: datetime,
                    due_window_hours: Optional[float] = None) -> str:
    """'overdue' if any open task is past due, 'due' if one falls due within the window, else 'none'."""
    if due_window_hours is None:
        due_window_hours = load_priority_weights()["followup"]["due_window_hours"]
    if not open_task_due_times:
        return "none"
    soonest = min(open_task_due_times)
    if soonest < now:
        return "overdue"
    if (soonest - now).total_seconds() / 3600 <= due_window_hours:
        return "due"
    return "none"
