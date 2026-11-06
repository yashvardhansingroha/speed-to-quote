"""Generate SYNTHETIC / DEMONSTRATION inquiry messages.

How variety is produced (so the text does not read like one template):
  1. A random *business profile* (name, states, fleet, cargo, vehicles, drivers ...).
  2. A *scenario*: the intent, how urgent, and how detailed ("sparse" / "medium" / "rich").
  3. Each checklist fact is included or skipped by probability that depends on the detail level,
     and is phrased by picking one of several wordings.
  4. A *voice* (casual, formal, terse, voicemail, sloppy-with-typos) rewrites the result.

The scenario (intent/urgency) is returned to the caller ONLY so the history simulator can behave
sensibly (e.g. only quote requests reach a quote). It is never stored and NEVER exported as a label
for the evaluation set: those labels are written by a human.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.data_generation import vocab as V
from src.extraction.textutils import _UNITS  # number words, for "twelve trucks"

NUMBER_WORDS = {v: k for k, v in _UNITS.items() if 1 <= v <= 20}


@dataclass
class Business:
    name: str
    first: str
    last: str
    descriptor: str
    cargo: str
    vehicle_types: list[str]
    states: list[str]
    fleet: int
    drivers: int
    experience: int
    usdot: int
    mc: int
    carrier: str
    new_venture: bool


@dataclass
class GeneratedInquiry:
    customer_name: str
    email: str
    phone: str
    channel: str
    message: str
    received_at: datetime
    # --- generator-internal scenario: used for event simulation only, never persisted ---
    scenario_intent: str = ""
    scenario_urgency: str = ""


# ----------------------------------------------------------------------------- helpers
def _pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def _fleet_size(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.28:
        return rng.randint(1, 3)
    if r < 0.65:
        return rng.randint(4, 12)
    if r < 0.90:
        return rng.randint(13, 30)
    return rng.randint(31, 75)


def make_business(rng: random.Random) -> Business:
    first, last = _pick(rng, V.FIRST_NAMES), _pick(rng, V.LAST_NAMES)
    entity = _pick(rng, V.BIZ_ENTITY)
    name = f"{_pick(rng, V.BIZ_PREFIX)} {_pick(rng, V.BIZ_SUFFIX)}" + (f" {entity}" if entity else "")
    desc, cargo, vehicles = _pick(rng, V.OPERATIONS)
    group = _pick(rng, V.STATE_GROUPS)
    states = rng.sample(group, rng.choice([1, 2, 2, 3, 3, 4]) if len(group) >= 4 else 2)
    fleet = _fleet_size(rng)
    drivers = max(1, fleet + rng.choice([-1, 0, 0, 1, 2, 4]) if fleet > 3 else rng.randint(1, fleet + 1))
    return Business(
        name=name, first=first, last=last, descriptor=desc, cargo=cargo, vehicle_types=vehicles,
        states=states, fleet=fleet, drivers=drivers, experience=rng.randint(2, 25),
        usdot=rng.randint(1_100_000, 3_900_000), mc=rng.randint(450_000, 1_150_000),
        carrier=_pick(rng, V.CARRIERS), new_venture=rng.random() < 0.12,
    )


def _num(rng: random.Random, n: int, words_ok: bool = True) -> str:
    if words_ok and n in NUMBER_WORDS and rng.random() < 0.3:
        return NUMBER_WORDS[n]
    return str(n)


def _states_phrase(rng: random.Random, states: list[str]) -> str:
    """'Texas and Oklahoma', 'TX/OK/AR', 'TX, OK and AR' ... (names, abbreviations or a mix)."""
    mode = rng.choice(["names", "names", "abbr", "mixed"])
    items = [V.ABBR[s] if mode == "abbr" or (mode == "mixed" and rng.random() < 0.5) else s for s in states]
    if len(items) == 1:
        return items[0]
    if rng.random() < 0.25:
        return "/".join(items)
    return ", ".join(items[:-1]) + (" and " if len(items) == 2 else ", and ") + items[-1]


def date_phrase(rng: random.Random, received: datetime, days: int) -> str:
    """A natural way to say 'this many days from the receipt date'."""
    d = received + timedelta(days=days)
    mon, day = d.strftime("%B"), d.day
    if days <= 0:
        return rng.choice(["today", "tonight", "right now"])
    if days == 1:
        return "tomorrow"
    if days <= 6:
        return rng.choice([f"in {days} days", f"by {d.strftime('%A')}", f"on {d.strftime('%A')}"])
    if days <= 14:
        return rng.choice([f"in {days // 7 or 1} week{'s' if days >= 14 else ''}", f"on {mon} {day}",
                           f"on {d.month}/{d.day}", f"in about {days} days"])
    if days <= 45:
        return rng.choice([f"on {mon} {day}", f"on {d.strftime('%b')} {day}", f"{d.month}/{d.day}",
                           "next month", f"in {round(days / 7)} weeks", "at the end of the month"])
    return rng.choice([f"in {mon}", f"on {mon} {day}", f"in {round(days / 30)} months", f"{d.month}/{d.day}"])


# ----------------------------------------------------------------------------- fact phrases
def _f_name(rng, b: Business) -> str:
    return rng.choice([
        f"This is {b.first} with {b.name}.", f"{b.name} here.", f"I own {b.name}.",
        f"We're {b.name}.", f"My company is {b.name}.", f"{b.first} {b.last}, {b.name}.",
    ])


def _f_states(rng, b: Business) -> str:
    s = _states_phrase(rng, b.states)
    return rng.choice([
        f"We operate in {s}.", f"Mostly running {s}.", f"Our lanes are {s}.", f"We run through {s}.",
        f"Based in {b.states[0]}, hauling in {s}.", f"{s} are our main states.",
    ])


def _f_fleet(rng, b: Business) -> str:
    n = _num(rng, b.fleet)
    one = b.fleet == 1
    return rng.choice([
        f"We have {n} {'truck' if one else 'trucks'}.", f"Running {n} {'power unit' if one else 'power units'}.",
        f"Fleet of {n}.", f"Currently {n} {'tractor' if one else 'tractors'} on the road.",
        f"It's {'just me and one truck' if one else f'{n} units total'}.", f"We operate {n} {'truck' if one else 'trucks'}.",
        f"{'One rig' if one else f'{n} rigs'} right now.",
    ])


def _f_fleet_vague(rng, b: Business) -> str:
    return rng.choice(["We have a few trucks.", "Small fleet, growing.", "Several trucks.", "A handful of trucks."])


def _f_cargo(rng, b: Business) -> str:
    return rng.choice([
        f"We haul {b.cargo}.", f"Mostly hauling {b.cargo}.", f"Our freight is {b.cargo}.",
        f"We move {b.cargo} mainly.", f"Primarily transporting {b.cargo}.",
    ])


def _f_vehicles(rng, b: Business) -> str:
    y1 = rng.randint(2012, 2021)
    y2 = min(2025, y1 + rng.randint(1, 5))
    make = _pick(rng, V.MAKES)
    t = _pick(rng, b.vehicle_types)
    return rng.choice([
        f"Units are mostly {y1}-{y2} {make}s.", f"We have {y1} {make} {V.MODELS[make]}s and a {y2} {_pick(rng, V.MAKES)}.",
        f"Equipment: {t}, model years {y1} to {y2}.", f"All {t} from {y1} through {y2}.",
        f"Mix of {y1} and {y2} {make} {V.MODELS[make]} trucks.",
    ])


def _f_vehicles_vague(rng, b: Business) -> str:
    return rng.choice([f"All {_pick(rng, b.vehicle_types)}.", f"Mostly {_pick(rng, b.vehicle_types)}.", "Mixed equipment."])


def _f_drivers(rng, b: Business) -> str:
    k, e = _num(rng, b.drivers), b.experience
    if b.drivers == 1:
        return rng.choice([f"I drive myself, {e} years in trucking.", f"Just me driving, {e}+ years of CDL experience."])
    return rng.choice([
        f"{k} drivers, all with {e}+ years of CDL experience.", f"We have {k} drivers averaging {e} years of experience.",
        f"{k} drivers, most have {e} years driving.", f"Drivers: {k}, between {max(1, e - 3)} and {e + 4} years experience.",
    ])


def _f_drivers_vague(rng, b: Business) -> str:
    return rng.choice([f"We have {b.drivers} drivers.", "Experienced drivers.", f"{b.drivers} drivers on payroll."]
                      if b.drivers > 1 else ["Just me driving.", "I'm the only driver."])


def _f_ids(rng, b: Business) -> str:
    return rng.choice([
        f"USDOT {b.usdot}, MC {b.mc}.", f"DOT# {b.usdot}.", f"Our DOT number is {b.usdot}.",
        f"MC number {b.mc}.", f"USDOT: {b.usdot}",
    ])


def _f_radius(rng, b: Business) -> str:
    return rng.choice([
        "We run local, mostly within 150 miles.", "Regional work.", "Long-haul, over the road.", "OTR, we're gone a week at a time.",
        f"Radius is about {rng.choice([50, 100, 250, 300, 500])} miles.", "All local deliveries.", "Strictly intrastate.",
    ])


def _f_coverages(rng, b: Business) -> str:
    c = _pick(rng, V.COVERAGE_PHRASES)
    return rng.choice([f"We need {c}.", f"Looking at {c}.", f"Coverage wanted: {c}.", f"Want to quote {c}."])


def _f_coverages_vague(rng, b: Business) -> str:
    return rng.choice([f"We need {_pick(rng, V.VAGUE_COVERAGE)}.", f"Just want {_pick(rng, V.VAGUE_COVERAGE)}."])


def _f_effective(rng, b: Business, received: datetime) -> str:
    days = rng.randint(5, 80)
    d = received + timedelta(days=days)
    return rng.choice([
        f"Need it effective {d.strftime('%B')} {d.day}.", f"Want coverage to start {d.month}/{d.day}.",
        f"Start date would be {d.strftime('%b')} {d.day}.", f"Coverage needs to be in place by {d.strftime('%B')} {d.day}.",
    ])


def _f_effective_vague(rng, b: Business) -> str:
    return rng.choice(["Need it ASAP.", "Whenever the current one ends.", "As soon as possible."])


def _f_current(rng, b: Business, expiry_phrase: Optional[str]) -> str:
    if b.new_venture:
        return rng.choice(["We have no current insurance.", "No prior coverage, we just got our authority."])
    base = rng.choice([f"Currently insured with {b.carrier}.", f"Our current policy is through {b.carrier}.",
                       f"We're with {b.carrier} right now."])
    return base


def _f_loss(rng, b: Business) -> str:
    if b.new_venture:
        return rng.choice(["First truck, brand new company.", "We just started up so no history yet."])
    return rng.choice([
        "Loss runs attached.", f"No claims in the last {rng.choice([2, 3, 5])} years.", "Clean loss history.",
        "Attached are our loss runs.", f"Zero claims over {rng.choice([3, 4, 5])} years.",
    ])


def _f_loss_vague(rng, b: Business) -> str:
    return rng.choice(["Loss runs available on request.", "We had a couple of small claims.", "Had a few claims a while back."])


# probability a fact is included: (sparse, medium, rich)
P = {
    "name": (0.25, 0.55, 0.85), "ids": (0.02, 0.18, 0.55), "states": (0.35, 0.75, 0.95),
    "radius": (0.05, 0.30, 0.60), "cargo": (0.20, 0.55, 0.85), "fleet": (0.50, 0.90, 1.0),
    "vehicles": (0.04, 0.25, 0.70), "drivers": (0.05, 0.30, 0.70), "coverages": (0.10, 0.40, 0.80),
    "effective": (0.05, 0.20, 0.50), "current": (0.25, 0.60, 0.80), "loss": (0.02, 0.20, 0.50),
}
DETAIL_INDEX = {"sparse": 0, "medium": 1, "rich": 2}


def _facts(rng: random.Random, b: Business, detail: str, received: datetime,
           expiry_phrase: Optional[str], scale: float = 1.0) -> list[str]:
    """Pick which facts appear and phrase each one (sometimes vaguely, on purpose)."""
    i = DETAIL_INDEX[detail]

    def has(key: str) -> bool:
        return rng.random() < P[key][i] * scale

    out: list[str] = []
    if has("name"): out.append(_f_name(rng, b))
    if has("states"): out.append(_f_states(rng, b))
    if has("fleet"): out.append(_f_fleet(rng, b) if rng.random() > 0.08 else _f_fleet_vague(rng, b))
    if has("cargo"): out.append(_f_cargo(rng, b))
    if has("radius"): out.append(_f_radius(rng, b))
    if has("vehicles"): out.append(_f_vehicles(rng, b) if rng.random() > 0.25 else _f_vehicles_vague(rng, b))
    if has("drivers"): out.append(_f_drivers(rng, b) if rng.random() > 0.3 else _f_drivers_vague(rng, b))
    if has("ids"): out.append(_f_ids(rng, b))
    if has("coverages"): out.append(_f_coverages(rng, b) if rng.random() > 0.2 else _f_coverages_vague(rng, b))
    if has("effective"): out.append(_f_effective(rng, b, received) if rng.random() > 0.3 else _f_effective_vague(rng, b))
    if has("current"): out.append(_f_current(rng, b, expiry_phrase))
    if has("loss"): out.append(_f_loss(rng, b) if rng.random() > 0.25 else _f_loss_vague(rng, b))
    return out


# ----------------------------------------------------------------------------- intent cores
def _timing(rng: random.Random, urgency: str, received: datetime) -> tuple[Optional[str], Optional[int]]:
    """(phrase, days) for the deadline matching the urgency level, or (None, None)."""
    if urgency == "high":
        days = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 9, 12])
    elif urgency == "medium":
        days = rng.choice([15, 18, 21, 25, 30, 35, 40])
    else:
        days = rng.choice([55, 70, 90, 120]) if rng.random() < 0.5 else None
    return (date_phrase(rng, received, days), days) if days is not None else (None, None)


def _core(rng: random.Random, intent: str, b: Business, urgency: str, received: datetime) -> tuple[str, Optional[str]]:
    """The sentence(s) that express the customer's main purpose. Returns (text, expiry_phrase)."""
    when, _ = _timing(rng, urgency, received)
    urg_tail = {"high": rng.choice([" Need to move fast on this.", " It's urgent.", " ASAP please.", ""]),
                "medium": rng.choice([" Hoping to get this done soon.", "", " Next couple weeks would be great."]),
                "low": rng.choice([" No rush.", " Just exploring options for now.", "", " Planning ahead."])}[urgency]
    co = rng.choice(["our trucking company", "my trucking business", "our fleet", "my trucks"])
    if intent == "new_quote":
        opts = [
            f"Looking to get a quote for {co}.", f"We need insurance for {co} and would like a quote.",
            f"Can you quote commercial truck coverage for us?", f"Need a quote on liability for {co}.",
            f"Shopping for trucking insurance, can someone reach out?", "Please send options and pricing for our rigs.",
            f"{b.name} is looking for trucking insurance.", f"I'd like a quote on insuring {b.fleet if b.fleet > 1 else 'a'} "
            f"{'trucks' if b.fleet > 1 else 'truck'}.",
        ]
        if b.new_venture:
            opts += [f"I'm starting up a trucking company and need coverage.", "Just got my authority and need insurance quotes."]
        text = _pick(rng, opts)
        if when and urgency != "low":
            text += f" Want coverage {when}."
        return text + urg_tail, None
    if intent == "renewal_or_replacement":
        w = when or "in a few months"
        opts = [
            f"Our policy expires {w} and we'd like to see what else is out there.",
            f"Insurance is up for renewal {w}. The renewal number from our carrier jumped a lot.",
            f"Our carrier non-renewed us, we need replacement coverage {w}.",
            f"We want to switch from {b.carrier} before the policy ends {w}.",
            f"Current policy ends {w}. Can you shop it around for us?",
            f"Renewal is coming up {w}; looking for a better option.",
            f"Policy lapses {w} and I haven't lined up anything new.",
        ]
        return _pick(rng, opts) + urg_tail, w
    if intent == "coverage_question":
        return _pick(rng, [
            "Do I need cargo coverage if I only haul my own equipment?",
            "What's the difference between primary liability and general liability for a trucking company?",
            "Is non-trucking liability required if my trucks are leased to a carrier?",
            "Does physical damage coverage apply when a trailer is detached?",
            "Quick question: are we required to carry more than the state minimum if we cross state lines?",
            "Wondering what is covered if a load gets stolen from a yard.",
            "Does a policy for a box truck business also cover a pickup we use for errands?",
        ]) + rng.choice(["", " Thanks.", " Not looking to buy right now, just trying to understand."]), None
    if intent == "document_or_info_provided":
        return _pick(rng, [
            "Attached are the documents you asked for.", "Here is the driver list and vehicle schedule you requested.",
            "As requested, sending over our loss runs and a list of units.", "Per your email, see attached dec page and driver info.",
            "Sending over the info you asked for. Let me know if anything is missing.",
            "Attaching our loss runs now. Hope that's what you needed.",
        ]), None
    if intent == "follow_up":
        return _pick(rng, [
            "Following up on my email from last week. Haven't heard back.", "Just checking in on the quote request I sent Tuesday.",
            "Any update on this? I called twice and left a message.", "Circling back, still waiting to hear from someone.",
            "Touching base again, we really need to get this moving.", "Second email, no response yet on my quote.",
        ]) + urg_tail, None
    if intent == "existing_policy_service":
        return _pick(rng, [
            "I need a certificate of insurance for a shipper by end of day.", "Need to add a new truck to my policy, how do I do that?",
            "Can you remove a driver from our policy? He left last month.", "Please send an updated COI naming our broker as additional insured.",
            "We need to update our mailing address on the policy.", "Looking for a copy of our insurance ID cards.",
            "I need to report a claim, a trailer was damaged at a yard.",
            f"Policy number {rng.randint(100000, 999999)}, need an endorsement for a new unit.",
        ]), None
    return "", None


# ----------------------------------------------------------------------------- voices
TYPOS = {
    "insurance": "insurence", "trucks": "trucs", "expires": "expirs", "policy": "polcy", "coverage": "coverge",
    "quote": "qoute", "liability": "liabilty", "with": "wiht", "the": "teh", "because": "becuase",
    "company": "compnay", "operating": "operatng", "business": "buisness", "please": "pleas", "current": "curent",
}


def _typos(rng: random.Random, text: str) -> str:
    def swap(m: re.Match) -> str:
        w = m.group(0)
        if w.lower() in TYPOS and rng.random() < 0.6:
            return TYPOS[w.lower()]
        if len(w) > 5 and rng.random() < 0.07:
            i = rng.randrange(1, len(w) - 2)
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        return w
    return re.sub(r"[A-Za-z]+", swap, text)


def _voice(rng: random.Random, body: str, b: Business, channel: str) -> str:
    style = rng.choices(["casual", "formal", "terse", "voicemail", "sloppy"], weights=[34, 18, 16, 10, 22])[0]
    greet = {
        "casual": rng.choice(["Hi,", "Hey,", "Hello,", "Hi there,", ""]),
        "formal": rng.choice(["Good afternoon,", "To whom it may concern,", "Hello,", "Good morning,"]),
        "terse": "", "voicemail": "", "sloppy": rng.choice(["hi", "hey guys", "hello", ""]),
    }[style]
    close = {
        "casual": rng.choice(["Thanks!", "Thanks,", "Appreciate it.", "Let me know.", ""]),
        "formal": rng.choice(["Thank you for your time.", "Kind regards,", "Please advise.", "Sincerely,"]),
        "terse": rng.choice(["Thx", "", "Call me.", "Thanks"]), "voicemail": "Okay, thanks, bye.",
        "sloppy": rng.choice(["thx", "thanks", "pls get back to me", ""]),
    }[style]
    sig = rng.choice([b.first, f"{b.first} {b.last}", f"- {b.first}", f"{b.first}\n{b.name}", ""]) if style != "voicemail" else ""
    if style == "formal":
        body = body.replace("We ", "We ", 1)
    if style == "terse":
        body = body.replace("Currently ", "").replace("Looking to get", "Need").replace("We would like", "Want")
    if style == "voicemail":
        body = f"Hi, this is {b.first}, uh, calling about insurance. " + body.replace(". ", ", um, ", 1) + " "
        body = "[Voicemail transcript] " + body
    if style == "sloppy":
        body = _typos(rng, body)
        if rng.random() < 0.4:
            body = body.lower()
    parts = [p for p in (greet, body.strip(), close, sig) if p]
    text = (" " if style in ("terse", "sloppy", "voicemail") else "\n\n").join(parts) if style != "formal" else "\n\n".join(parts)
    if channel == "sms" and len(text) > 220:                  # texts are short: cut at a sentence end
        cut = text[:220]
        text = cut[:cut.rfind(". ") + 1] if ". " in cut else cut.rsplit(" ", 1)[0]
    return text.strip()


# ----------------------------------------------------------------------------- top level
def _received_at(rng: random.Random, now: datetime, history_days: int) -> datetime:
    """Random timestamp in the history window, weighted to weekday business hours."""
    while True:
        day = now - timedelta(days=rng.random() * history_days)
        if day.weekday() >= 5 and rng.random() < 0.7:
            continue                                   # fewer weekend inquiries
        hour = int(rng.triangular(6, 21, 10.5))
        t = day.replace(hour=hour, minute=rng.randint(0, 59), second=0, microsecond=0)
        if t < now - timedelta(minutes=5):
            return t


def generate_inquiry(rng: random.Random, now: datetime, history_days: int) -> GeneratedInquiry:
    b = make_business(rng)
    received = _received_at(rng, now, history_days)
    channel = rng.choices(["email", "web_form", "phone_note", "sms"], weights=[55, 25, 12, 8])[0]
    kind = rng.choices(
        ["new_quote", "renewal_or_replacement", "coverage_question", "document_or_info_provided",
         "follow_up", "existing_policy_service", "short", "ambiguous", "multi"],
        weights=[27, 19, 7, 8, 8, 8, 8, 4, 11])[0]
    urgency = rng.choices(["high", "medium", "low"], weights=[28, 42, 30])[0]
    phone = f"555-01{rng.randint(0, 99):02d}"
    email = f"{b.first.lower()}.{b.last.lower()}@example.com"

    intent = kind
    if kind in ("short", "ambiguous"):
        pool = V.SHORT_MESSAGES if kind == "short" else V.AMBIGUOUS_MESSAGES
        text = _pick(rng, pool).format(first=b.first, phone=phone)
        if rng.random() < 0.3:
            text = text.lower()
        intent = "unclear" if kind == "ambiguous" or rng.random() < 0.6 else "new_quote"
        urgency = "high" if re.search(r"asap|urgent", text, re.I) else "low"
    elif kind == "multi":
        a, c = rng.sample(["new_quote", "renewal_or_replacement", "existing_policy_service", "coverage_question",
                           "follow_up", "document_or_info_provided"], 2)
        ta, expiry = _core(rng, a, b, urgency, received)
        tc, _ = _core(rng, c, b, urgency, received)
        detail = rng.choice(["sparse", "medium"])
        facts = _facts(rng, b, detail, received, expiry, scale=0.6)
        text = _voice(rng, " ".join([ta, rng.choice(["Also,", "And", "On another note,", "Separately,"]), tc[0].lower() + tc[1:], *facts]),
                      b, channel)
        intent = a
    else:
        detail = rng.choices(["sparse", "medium", "rich"], weights=[30, 45, 25])[0]
        core, expiry = _core(rng, kind, b, urgency, received)
        scale = 1.0 if kind in ("new_quote", "renewal_or_replacement") else 0.45
        facts = _facts(rng, b, detail, received, expiry, scale=scale)
        # new-business style messages lead with the purpose; some lead with who they are
        order = [core, *facts] if rng.random() < 0.7 else [*facts[:1], core, *facts[1:]]
        if rng.random() < 0.25 and len(order) > 3:
            tail = order[1:]
            rng.shuffle(tail)
            order = [order[0], *tail]
        if urgency == "high" and kind in ("new_quote", "renewal_or_replacement") and rng.random() < 0.4:
            order.append(rng.choice(["Can someone call me today?", "Please call me this week.", "Need to talk to someone ASAP."]))
        if rng.random() < 0.35:                                   # a little irrelevant context
            order.insert(rng.randint(1, len(order)), _pick(rng, V.FILLER).format(year=rng.randint(2003, 2023)))
        if rng.random() < 0.25:
            order.append(rng.choice([f"My number is {phone}.", f"Call me at {phone}.", f"You can reach me at {email}."]))
        text = _voice(rng, " ".join(order), b, channel)

    return GeneratedInquiry(
        customer_name=f"{b.first} {b.last}", email=email, phone=phone, channel=channel,
        message=text, received_at=received, scenario_intent=intent, scenario_urgency=urgency)


def generate_inquiries(n: int, seed: int, now: datetime, history_days: int) -> list[GeneratedInquiry]:
    """Reproducible: the same (n, seed, now, history_days) always yields identical output."""
    rng = random.Random(seed)
    out = [generate_inquiry(rng, now, history_days) for _ in range(n)]
    out.sort(key=lambda g: g.received_at)
    return out
