"""Small shared text helpers: number words, US states, whitespace normalisation."""
from __future__ import annotations

import re
from typing import Optional

# --------------------------------------------------------------------------- numbers
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_TENS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
_ONES = {w for w, v in _UNITS.items() if 1 <= v <= 9}

# Words understood in messages but deliberately NOT part of _UNITS, because the synthetic-data generator builds its
# own wording from _UNITS and must not change (the evaluation messages are generated from it).
_EXTRA_WORDS = {"dozen": 12, "single": 1}
_WORD_ALT = "|".join(sorted({**_UNITS, **_EXTRA_WORDS}, key=len, reverse=True))
# Regex for "12", "1,200", "twelve" or "twenty-five" / "twenty five". Non-capturing on purpose.
NUM_PATTERN = rf"(?:\d{{1,3}}(?:,\d{{3}})+|\d+|(?:{'|'.join(sorted(_TENS))})[- ](?:{'|'.join(sorted(_ONES))})|{_WORD_ALT})"


def parse_number(text: str) -> Optional[int]:
    """'12' -> 12, 'twelve' -> 12, 'twenty-five' -> 25, 'a' -> None."""
    t = text.strip().lower().replace(",", "")
    if t.isdigit():
        return int(t)
    if t in _UNITS:
        return _UNITS[t]
    if t in _EXTRA_WORDS:
        return _EXTRA_WORDS[t]
    parts = re.split(r"[- ]", t)
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES:
        return _UNITS[parts[0]] + _UNITS[parts[1]]
    return None


# --------------------------------------------------------------------------- US states
STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY",
}
STATE_ABBRS = set(STATE_NAMES.values())
# Abbreviations that are also ordinary English words. Only accepted next to another state.
AMBIGUOUS_ABBRS = {"IN", "OR", "ME", "OK", "HI", "OH", "ID"}


# --------------------------------------------------------------------------- normalisation
def collapse_ws(text: str) -> str:
    """Collapse runs of whitespace (spaces, tabs, newlines) into single spaces. NOTHING else.

    This is the only leniency in quote checking: a model cannot reproduce the exact line breaks
    of a pasted message, so a quote that spans a line break still matches. Case, punctuation,
    apostrophes and every letter must still match exactly.
    """
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- fleet counts
# "12 trucks", "twelve power units", "fleet of 12": a number tied to VEHICLE wording.
# Used by both the rules baseline and the evidence check, so "99 drivers" never counts as a fleet size.
FLEET_NOUNS = (r"(?:power units?|tractor[- ]trailers?|tractors?|box trucks?|dump trucks?|flatbed trucks?|"
               r"flatbeds?(?!\s+trailers?)|semi[- ]?trucks?|semis?|trucks?|trks?|rigs?|units?|vehicles?|reefers?|"
               r"vans?|pickups?|tankers?|hotshots?)")
# A number must not be glued to other digits or punctuation ("12.5 trucks", "555-0142 trucks", "$50,000 truck"), and the
# words between it and the noun must be real words ("5 2020 trucks" is not a count). "12-truck fleet" and "12+ trucks" work.
_NUM_START = r"(?<![\d.,/$#-])\b"
_FILLER = r"(?:[A-Za-z][\w-]*[\s-]+)"
FLEET_RX = re.compile(
    rf"{_NUM_START}(?P<n>{NUM_PATTERN})\+?[\s-]+{_FILLER}{{0,2}}?{FLEET_NOUNS}\b"
    rf"|\bfleet\s+(?:size\s+)?(?:of|is|:)\s*(?:about\s+|around\s+)?(?P<n2>{NUM_PATTERN})\b", re.IGNORECASE)
# "10 to 12 trucks", "10-12 trucks": a range is not one number.
FLEET_RANGE_RX = re.compile(
    rf"{_NUM_START}(\d+)\s*(?:-|\u2013|\u2014|to)\s*(\d+)\s+{_FILLER}{{0,2}}?{FLEET_NOUNS}\b", re.IGNORECASE)
# "over 20 trucks", "at least 5 trucks": a bound, not an exact count.
FLEET_BOUND_RX = re.compile(r"\b(?:over|more than|at least|under|less than|up to|fewer than|almost|nearly)\s+$", re.IGNORECASE)
