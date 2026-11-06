"""Submission completeness.

    completeness = fields with status `provided` / fields that APPLY (everything except `not_applicable`)

`unclear` and `missing` both count as not yet supplied. This number means "how ready is this for a
HUMAN to review", NOT "ready for a bindable quote": carriers decide what they need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

READINESS_NOTE = ("Completeness means ready for human review, NOT ready for a bindable quote. "
                  "Carriers and underwriters decide what they actually require.")


@dataclass(frozen=True)
class Completeness:
    provided: int
    applicable: int
    total: int
    missing: list[str] = field(default_factory=list)       # field keys, checklist order
    unclear: list[str] = field(default_factory=list)
    not_applicable: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        """0.0 when nothing applies (avoids dividing by zero)."""
        return self.provided / self.applicable if self.applicable else 0.0

    @property
    def text(self) -> str:
        return f"Submission completeness: {self.provided} of {self.applicable} applicable fields supplied."


def compute_completeness(statuses: Mapping[str, str]) -> Completeness:
    """`statuses` maps checklist field key -> provided | missing | unclear | not_applicable."""
    missing, unclear, na, provided = [], [], [], 0
    for key, status in statuses.items():
        if status == "provided":
            provided += 1
        elif status == "not_applicable":
            na.append(key)
        elif status == "unclear":
            unclear.append(key)
        elif status == "missing":
            missing.append(key)
        else:
            raise ValueError(f"unknown checklist status {status!r} for {key}")
    return Completeness(provided=provided, applicable=len(statuses) - len(na), total=len(statuses),
                        missing=missing, unclear=unclear, not_applicable=na)
