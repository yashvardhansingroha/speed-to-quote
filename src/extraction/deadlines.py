"""Turning date phrases ("next month", "by Friday", "Dec 5") into dates - deterministically.

Why this exists: an LLM is good at *finding* "our policy expires next month" but unreliable at
calendar arithmetic. So the LLM only returns the customer's own words (verified verbatim), and
THIS module converts the words into dates relative to when the message was received.

Three safeguards against false precision / false deadlines:
  1. RANGES: a vague phrase ("next month", "in March") resolves to a range (earliest, latest) and is
     flagged `approximate`. Scoring uses the EARLIEST plausible day (so a possibly-imminent expiry
     never looks distant) and the UI says "approximate: confirm the exact date".
  2. CUES: an absolute date ("March 3", "12/15", "on Tuesday", "in March") only counts as a
     deadline if a deadline word ("expires", "effective", "need ... by", "call me" ...) comes before
     it in the same sentence and is closer than any past-tense word ("bought", "called", "started").
     "We bought 3 trucks on March 3, 2020" therefore yields no deadline.
  3. STALENESS: a date that ends more than 30 days before the message was received is ignored.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Optional

from src.extraction.textutils import NUM_PATTERN, parse_number

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
            "saturday": 5, "sunday": 6}

_MONTH_FULL = "january|february|march|april|may|june|july|august|september|october|november|december"
_MONTH_ABBR = "jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
_MONTH_ANY = f"(?:{_MONTH_FULL}|{_MONTH_ABBR})"
_MONTH_NOT_MAY = f"(?:{_MONTH_FULL.replace('may|', '')}|{_MONTH_ABBR})"
_WEEKDAY = "|".join(WEEKDAYS)
STALE_DAYS = 30

Range = tuple[date, date]          # (earliest, latest)


@dataclass(frozen=True)
class DateMatch:
    text: str                      # the exact words found in the message
    earliest: date                 # earliest day consistent with the words
    latest: date                   # latest day consistent with the words
    start: int
    end: int

    @property
    def date(self) -> date:
        """The day used for scoring: the earliest plausible one."""
        return self.earliest

    @property
    def approximate(self) -> bool:
        return self.earliest != self.latest


# --------------------------------------------------------------------------- calendar helpers
def _last_day(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _first_of_next_month(d: date) -> date:
    return _last_day(d.year, d.month) + timedelta(days=1)


def _add_months(d: date, n: int) -> date:
    idx = d.month - 1 + n
    y, m = d.year + idx // 12, idx % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _friday_on_or_after(d: date) -> date:
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _monday_next_week(d: date) -> date:
    return d + timedelta(days=7 - d.weekday())


def _exact(d: date) -> Range:
    return d, d


# --------------------------------------------------------------------------- resolvers
# Each takes (regex match, received date) and returns (earliest, latest) or None.
def _r_relative_in(m: re.Match, today: date) -> Optional[Range]:
    raw = m.group("n").lower()
    n = 2 if "couple" in raw else (1 if raw in ("a", "an") else parse_number(raw))
    if n is None:
        return None
    unit = m.group("unit").lower()
    if unit == "day":
        return _exact(today + timedelta(days=n))
    if unit == "week":
        return _exact(today + timedelta(weeks=n))
    return _exact(_add_months(today, n))


def _r_today(m: re.Match, today: date) -> Optional[Range]:
    return _exact(today + timedelta(days=1) if m.group("w").lower() == "tomorrow" else today)


def _r_this_next(m: re.Match, today: date) -> Optional[Range]:
    which, unit = m.group("w").lower(), m.group("u").lower()
    if unit == "week":     # a "by the end of the week" request: Friday is the deadline, not a guess
        return _exact(_friday_on_or_after(today) if which == "this"
                      else _monday_next_week(today) + timedelta(days=4))
    if which == "this":    # sometime between now and month end
        return today, _last_day(today.year, today.month)
    first = _first_of_next_month(today)
    return first, _last_day(first.year, first.month)


def _r_end_of(m: re.Match, today: date) -> Optional[Range]:
    mod = (m.group("m") or "").strip().lower()
    if m.group("u").lower() == "week":
        return _exact(_friday_on_or_after(today) if mod != "next"
                      else _monday_next_week(today) + timedelta(days=4))
    base = today if mod != "next" else _first_of_next_month(today)
    return _exact(_last_day(base.year, base.month))


def _r_weekday(m: re.Match, today: date) -> Optional[Range]:
    target = WEEKDAYS[m.group("d").lower()]
    if m.group("p").lower() == "next":
        return _exact(_monday_next_week(today) + timedelta(days=target))
    return _exact(today + timedelta(days=(target - today.weekday()) % 7))


def _roll_year(month: int, day: int, year: Optional[int], today: date) -> Optional[date]:
    """Month/day with no year: pick the occurrence that makes sense relative to `today`."""
    try:
        if year is not None:
            return date(year, month, day)          # explicit year: take it literally
        cand = date(today.year, month, day)
        if cand < today - timedelta(days=STALE_DAYS):
            cand = date(today.year + 1, month, day)
        return cand
    except ValueError:
        return None


def _r_month_day(m: re.Match, today: date) -> Optional[Range]:
    year = int(m.group("y")) if m.group("y") else None
    d = _roll_year(MONTHS[m.group("mon").lower()], int(m.group("d")), year, today)
    return _exact(d) if d else None


def _r_numeric(m: re.Match, today: date) -> Optional[Range]:
    mo, dd = int(m.group("m")), int(m.group("d"))
    if not (1 <= mo <= 12 and 1 <= dd <= 31):
        return None
    y = m.group("y")
    year = (2000 + int(y) if len(y) == 2 else int(y)) if y else None
    d = _roll_year(mo, dd, year, today)
    return _exact(d) if d else None


def _r_iso(m: re.Match, today: date) -> Optional[Range]:
    d = _roll_year(int(m.group("mo")), int(m.group("d")), int(m.group("y")), today)
    return _exact(d) if d else None


def _r_business_days(m: re.Match, today: date) -> Optional[Range]:
    n = parse_number(m.group("n"))
    if n is None or n > 365:
        return None
    d = today
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return _exact(d)


def _r_fortnight(m: re.Match, today: date) -> Optional[Range]:
    return _exact(today + timedelta(days=14))


def _r_quarter_end(m: re.Match, today: date) -> Optional[Range]:
    return _exact(_last_day(today.year, 3 * int(m.group("q"))))


def _r_day_ordinal(m: re.Match, today: date) -> Optional[Range]:
    """"on the 15th": the next 15th on or after today."""
    day = int(m.group("d"))
    for months_ahead in (0, 1, 2):
        base = _add_months(today.replace(day=1), months_ahead)
        try:
            cand = date(base.year, base.month, day)
        except ValueError:
            continue
        if cand >= today:
            return _exact(cand)
    return None


def _r_month_only(m: re.Match, today: date) -> Optional[Range]:
    mon = MONTHS[m.group("mon").lower()]
    year = today.year if mon >= today.month else today.year + 1
    first, last = date(year, mon, 1), _last_day(year, mon)
    return max(first, today), last                    # approximate: anywhere in that month


# (name, regex with named groups, resolver, absolute?)  absolute => needs a deadline cue.
_N = NUM_PATTERN
_RULES: list[tuple[str, str, Callable, bool]] = [
    ("relative_in",
     rf"\bin\s+(?:about\s+|around\s+|roughly\s+)?(?P<n>(?:a\s+)?couple(?:\s+of)?|{_N}|an?)\s+(?P<unit>day|week|month)s?\b",
     _r_relative_in, False),
    ("today", r"\b(?P<w>today|tonight|tomorrow)\b", _r_today, False),
    ("end_of", r"\bend\s+of\s+(?P<m>the\s+|this\s+|next\s+)?(?P<u>week|month)\b", _r_end_of, False),
    ("this_next", r"\b(?P<w>this|next)\s+(?P<u>week|month)\b", _r_this_next, False),
    ("weekday", rf"\b(?P<p>by|on|before|until|till|this|next|for|call me|text me|reach me|ring me|talk|speak|meet|call)\s+(?P<d>{_WEEKDAY})\b",
     _r_weekday, True),
    ("iso", r"\b(?P<y>20\d{2})-(?P<mo>\d{1,2})-(?P<d>\d{1,2})\b", _r_iso, True),
    ("dash_mdy", r"(?<![\d-])(?P<m>\d{1,2})-(?P<d>\d{1,2})-(?P<y>\d{4}|\d{2})(?![\d-])", _r_numeric, True),
    ("business_days", rf"\bin\s+(?:about\s+|around\s+)?(?P<n>{_N})\s+business\s+days?\b", _r_business_days, False),
    ("fortnight", r"\bin\s+(?:a|one|1)\s+fortnight\b", _r_fortnight, False),
    ("quarter_end", r"\bend\s+of\s+(?:the\s+)?Q(?P<q>[1-4])\b", _r_quarter_end, True),
    ("day_ordinal", r"\bthe\s+(?P<d>\d{1,2})(?:st|nd|rd|th)\b", _r_day_ordinal, True),
    ("month_day",
     rf"\b(?P<mon>{_MONTH_ANY})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{{4}}))?\b",
     _r_month_day, True),
    ("day_month",
     rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon>{_MONTH_NOT_MAY})\b(?:,?\s+(?P<y>\d{{4}}))?",
     _r_month_day, True),
    ("numeric",
     r"(?<![\d/])(?P<m>\d{1,2})/(?P<d>\d{1,2})(?:/(?P<y>\d{4}|\d{2}))?(?![\d/])(?!\s*(?:ton|inch|in\b|%|\"))",
     _r_numeric, True),
    ("month_only", rf"\b(?P<p>in|during|end\s+of)\s+(?P<mon>{_MONTH_FULL})\b", _r_month_only, True),
]
_COMPILED = [(n, re.compile(src, re.IGNORECASE), fn, absolute) for n, src, fn, absolute in _RULES]

# Words that make a date a deadline / appointment ...
_FUTURE_CUE = re.compile(
    r"expir\w*|renew\w*|effective|\bstart(?:s|ing)?\b|\bbegin\w*|\bbind\w*|\bneed\w*|\bwant\w*|deadline|"
    r"\bdue\b|\buntil\b|\btill\b|\bbefore\b|\bby\b|\bends?\b|\bending\b|\blapse[sd]?\b|runs? out|up for|"
    r"\bcall\b|\btalk\b|\bspeak\b|\bmeet\b|\breach\b|in place|\bready\b|contact|get back|schedule",
    re.IGNORECASE)
# ... and words that make it history.
_PAST_CUE = re.compile(
    r"\b(?:called|emailed|e-mailed|sent|spoke|talked|left|wrote|submitted|messaged|texted|mentioned|"
    r"applied|ago|last|since|yesterday|earlier|bought|purchased|started|founded|established|opened|"
    r"formed|acquired|began|got|joined|hired|had)\b", re.IGNORECASE)


# A sentence ends at . ! ? or a blank line. A single line break is NOT a boundary: pasted emails
# often wrap mid-sentence ("Our policy expires\nDecember 5").
_BOUNDARY = re.compile(r"[.!?]|\n\s*\n")


def _context_before(text: str, start: int, window: int = 80) -> str:
    """Text before the match, limited to the same sentence (and `window` characters)."""
    seg = text[max(0, start - window): start]
    ends = [m.end() for m in _BOUNDARY.finditer(seg)]
    return seg[max(ends):] if ends else seg


def _last_end(rx: re.Pattern, s: str) -> int:
    ends = [m.end() for m in rx.finditer(s)]
    return max(ends) if ends else -1


# "not urgent", "isn't urgent", "don't need it today": the time word is being denied, not requested.
NEGATION_RX = re.compile(
    r"(?:\bnot|\bno longer|n['’]t|\bnever|\bwithout|\bhardly)\s+(?:\w+\s+){0,2}$", re.IGNORECASE)

_BUSINESS_NAME_TAIL = re.compile(
    r"['\u2019]?s?\s+(?:transport\w*|trucking|logistics|freight|hauling|express|carriers?|cartage|lines|llc|inc|co|corp|"
    r"brothers|bros|enterprises|services|industries)\b", re.IGNORECASE)

# Cues that may FOLLOW a date: "November 6 is our policy expiry date", "March 3 is the deadline".
_AFTER_CUE = re.compile(r"expir\w*|renew\w*|\blapse[sd]?\b|deadline|\bdue\b|\bends?\b|runs? out", re.IGNORECASE)


def _context_after(text: str, end: int, window: int = 60) -> str:
    """Text after the match, limited to the same sentence (and `window` characters)."""
    seg = text[end: end + window]
    m = _BOUNDARY.search(seg)
    return seg[: m.start()] if m else seg


def is_deadline_context(text: str, start: int, end: Optional[int] = None, absolute: bool = False) -> bool:
    """Does the wording around text[start:end] make it a forward-looking deadline / request?

    absolute=True (dates like "March 3", "on Tuesday"): needs a deadline word before it that is closer
    than any past-tense word, OR a deadline noun right after it in the same sentence ("... is our
    policy expiry date"). absolute=False (words like "today", "next month"): fine unless a
    past-tense word is the closest cue ("we bought a truck today").
    """
    if not absolute and end is not None and _BUSINESS_NAME_TAIL.match(text[end:end + 30]):
        return False                          # "Today Transport LLC", "Tomorrow's Logistics"
    if not absolute and NEGATION_RX.search(text[max(0, start - 30):start]):
        return False                          # "we don't need it today"
    before = _context_before(text, start)
    future, past = _last_end(_FUTURE_CUE, before), _last_end(_PAST_CUE, before)
    if absolute:
        if future >= 0 and future > past:
            return True
        after = _context_after(text, end if end is not None else start)
        return bool(_AFTER_CUE.search(after)) and not _PAST_CUE.search(after)
    return past < 0 or future > past


def _macro_source() -> str:
    """The <DATE> token used inside checklist regexes: every date form, without group names."""
    parts = [re.sub(r"\(\?P<\w+>", "(?:", src) for _, src, _, _ in _RULES]
    return "(?:" + "|".join(parts) + ")"


DATE_PATTERN = _macro_source()


# --------------------------------------------------------------------------- public API
def find_dates(text: str, received_at: datetime) -> list[DateMatch]:
    """All deadline-like date phrases in `text`, resolved relative to `received_at`."""
    today = received_at.date()
    found: list[DateMatch] = []
    for _, rx, resolver, absolute in _COMPILED:
        for m in rx.finditer(text):
            if not is_deadline_context(text, m.start(), m.end(), absolute):
                continue
            rng = resolver(m, today)
            if rng is None or rng[1] < today - timedelta(days=STALE_DAYS):
                continue                                # ignore dates long in the past
            found.append(DateMatch(m.group(0), rng[0], rng[1], m.start(), m.end()))

    # If two matches overlap ("end of next month" also contains "next month"), keep the longer one.
    found.sort(key=lambda d: -(d.end - d.start))
    kept: list[DateMatch] = []
    for d in found:
        if not any(d.start < k.end and k.start < d.end for k in kept):
            kept.append(d)
    return sorted(kept, key=lambda d: d.start)


def find_deadline(text: str, received_at: datetime) -> Optional[DateMatch]:
    """The match with the EARLIEST earliest-date = the most urgent one."""
    matches = find_dates(text, received_at)
    return min(matches, key=lambda d: d.earliest) if matches else None


def resolve_deadline_range(texts: list[str], received_at: datetime) -> Optional[tuple[date, date, str]]:
    """Resolve a deadline from the extracted quote / value: (earliest, latest, matched words)."""
    best: Optional[DateMatch] = None
    for t in texts:
        if not t or t == "not_provided":
            continue
        hit = find_deadline(t, received_at)
        if hit and (best is None or hit.earliest < best.earliest):
            best = hit
    return (best.earliest, best.latest, best.text) if best else None


def _locate(text: str, quote: str) -> Optional[tuple[int, int]]:
    """Where a (verified) quote sits in the message; tolerant only of whitespace differences."""
    q = (quote or "").strip()
    if not q or q == "not_provided":
        return None
    i = text.find(q)
    if i >= 0:
        return i, i + len(q)
    m = re.search(r"\s+".join(re.escape(tok) for tok in q.split()), text)
    return (m.start(), m.end()) if m else None


def resolve_deadline_in_message(text: str, quotes: list[str],
                                received_at: datetime) -> Optional[tuple[date, date, str]]:
    """Resolve the deadline using the WHOLE message for context.

    An LLM may quote only "December 5" while the message says "Our policy expires December 5".
    Judging the bare quote would lose the word "expires", so instead we find every deadline-like
    date in the full message and keep those that overlap the quoted words.
    Returns (earliest, latest, matched words) for the most urgent overlapping match, or None.
    """
    spans = [s for q in quotes if (s := _locate(text, q))]
    hits = [m for m in find_dates(text, received_at)
            if any(m.start < e and s < m.end for s, e in spans)]
    if not hits:
        return None
    best = min(hits, key=lambda m: m.earliest)
    return best.earliest, best.latest, best.text


def resolve_deadline_date(texts: list[str], received_at: datetime) -> Optional[date]:
    """Same as resolve_deadline_range but only the scoring (earliest) date."""
    r = resolve_deadline_range(texts, received_at)
    return r[0] if r else None


def clause_around(text: str, match: DateMatch, max_len: int = 160) -> str:
    """A verbatim snippet of the sentence containing the date phrase (used as evidence)."""
    starts = [text.rfind(c, 0, match.start) for c in ".!?\n"]
    s = max(starts) + 1
    ends = [i for i in (text.find(c, match.end) for c in ".!?\n") if i != -1]
    e = min(ends) if ends else len(text)
    if e - s > max_len:
        s, e = max(s, match.start - 80), min(e, match.end + 60)
    return text[s:e].strip(" ,;:-—")
