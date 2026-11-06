"""Regex helpers shared by the rules baseline and the evidence checks."""
from __future__ import annotations

import re

from src.extraction.deadlines import DATE_PATTERN
from src.extraction.textutils import NUM_PATTERN


def expand(pattern: str, macros: dict[str, str]) -> str:
    """Replace <NUM>, <DATE> and any config macro (<COVERAGE_TERM> ...) inside a regex."""
    out = pattern.replace("<NUM>", NUM_PATTERN).replace("<DATE>", DATE_PATTERN)
    for name, src in macros.items():
        out = out.replace(f"<{name}>", src)
    return out


def compile_patterns(patterns: list[str], macros: dict[str, str]) -> list[re.Pattern]:
    return [re.compile(expand(p, macros), re.IGNORECASE) for p in patterns]
