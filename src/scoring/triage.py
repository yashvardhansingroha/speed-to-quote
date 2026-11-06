"""Triage = decide whether to score an inquiry or send it to a human, then score it.

Routing rule (from the spec): if the intent is `unclear`, or the message is empty / too short,
the inquiry goes to human review and is NOT given a priority score. We also route to review
when the quote supporting the urgency label was invented (the label cannot be trusted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from src.extraction.deadlines import resolve_deadline_in_message, resolve_deadline_range
from src.extraction.pipeline import is_too_long, is_too_short
from src.extraction.schema import NOT_PROVIDED, InquiryExtraction
from src.scoring.priority import PriorityInputs, PriorityResult, compute_priority


@dataclass
class TriageResult:
    needs_review: bool
    review_reasons: list[str] = field(default_factory=list)
    priority: Optional[PriorityResult] = None      # None when routed to human review
    deadline_date: Optional[date] = None           # earliest plausible day (used for scoring)
    deadline_latest: Optional[date] = None         # latest plausible day
    deadline_approximate: bool = False
    days_until_deadline: Optional[int] = None
    hours_waiting: Optional[float] = None
    hints: list[str] = field(default_factory=list)  # things a human should clarify (not blockers)


def resolve_deadline(ex: InquiryExtraction, received_at: datetime,
                     text: Optional[str] = None) -> Optional[tuple[date, date]]:
    """Convert the (verified) deadline words into (earliest, latest) dates, relative to receipt.

    With the full message `text`, the quote is judged in context ("expires" + "December 5"),
    which is what the pipeline always does. Without it we can only read the quote on its own.
    """
    if ex.deadline.status != "provided":
        return None
    quotes = [ex.deadline.evidence, ex.deadline.value or ""]
    r = (resolve_deadline_in_message(text, quotes, received_at) if text is not None
         else resolve_deadline_range(quotes, received_at))
    return (r[0], r[1]) if r else None


def triage_inquiry(text: str, ex: InquiryExtraction, received_at: datetime, now: datetime,
                   first_response_at: Optional[datetime] = None,
                   followup: str = "none",
                   weights: Optional[dict[str, Any]] = None) -> TriageResult:
    """Route and (if appropriate) score one inquiry.

    `followup` is 'overdue' | 'due' | 'none' (see scoring.priority.followup_status).
    """
    reasons: list[str] = []
    if is_too_short(text):
        reasons.append("Message is empty or too short to analyse reliably.")
    if is_too_long(text):
        reasons.append("Message is longer than the automatic-analysis limit, so a person should read it.")
    if ex.intent == "unclear":
        reasons.append("Intent could not be determined." + (f" {ex.intent_note}" if ex.intent_note else ""))
    if ex.urgency_evidence_verified is False:
        reasons.append("The quote supporting the urgency label is missing from the message or describes a past event, so urgency is unreliable.")

    span = resolve_deadline(ex, received_at, text)
    deadline_date, latest = span if span else (None, None)
    approximate = bool(span and span[0] != span[1])
    days = (deadline_date - now.date()).days if deadline_date else None
    waiting = None if first_response_at else max(0.0, (now - received_at).total_seconds() / 3600)
    hints = []
    if approximate:
        hints.append(f"Deadline is approximate ({deadline_date:%b %d} to {latest:%b %d}): "
                     "ask the customer for the exact date.")

    if reasons:
        return TriageResult(True, reasons, None, deadline_date, latest, approximate, days, waiting, hints)

    # Prefer the short value ("next month") over the whole evidence sentence in the explanation.
    label = ex.deadline.value or (ex.deadline.evidence if ex.deadline.evidence != NOT_PROVIDED else None)
    result = compute_priority(
        PriorityInputs(urgency=ex.urgency, days_until_deadline=days, deadline_label=label,
                       deadline_approximate=approximate, hours_waiting=waiting, followup_status=followup),
        weights)
    return TriageResult(False, [], result, deadline_date, latest, approximate, days, waiting, hints)
