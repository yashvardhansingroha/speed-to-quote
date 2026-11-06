"""Follow-up draft generation (a HUMAN reviews and approves every draft; nothing is ever sent).

Two generators, one safety net:
  * template_draft  - deterministic, offline, built from the list of missing/unclear fields.
  * llm_draft       - the LLM returns STRUCTURED parts (which fields each group asks about + wording);
                      code checks the structure and composes the final text.
  * find_violations - deterministic scan applied to every draft (and to human edits, as warnings):
        no premiums/prices, coverage promises, eligibility talk, time commitments, or dates that the
        customer did not write themselves.

Rules the drafts must follow: use only information from the inquiry; ask only for fields that are
missing or unclear; never ask again for something already provided; group questions sensibly.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

from src.config import Checklist, ChecklistField
from src.extraction.deadlines import DATE_PATTERN
from src.extraction.llm_client import LLMCallError, LLMClient
from src.extraction.schema import InvalidLLMResponse, parse_llm_json

logger = logging.getLogger("speed_to_quote.drafting")

PLACEHOLDER_SIGNATURE = "[Your name]"


@dataclass
class DraftContext:
    """Everything a draft may use. All of it comes from the inquiry (or the human's corrections)."""
    message: str
    customer_first_name: Optional[str]
    checklist: Checklist
    statuses: Mapping[str, str]                           # field key -> provided|missing|unclear|not_applicable
    evidence: Mapping[str, str] = field(default_factory=dict)
    evidence_verified: Mapping[str, Optional[bool]] = field(default_factory=dict)
    deadline_words: Optional[str] = None                  # the customer's own words, if a deadline was found
    deadline_approximate: bool = False

    def requested(self) -> list[ChecklistField]:
        """Fields still needed: missing or unclear (checklist order)."""
        return [f for f in self.checklist.fields if self.statuses.get(f.key) in ("missing", "unclear")]

    def provided_fields(self) -> list[ChecklistField]:
        return [f for f in self.checklist.fields if self.statuses.get(f.key) == "provided"]


@dataclass
class DraftResult:
    body: str
    generated_by: str                                     # "llm" | "template"
    note: Optional[str] = None                            # why the template was used instead of the LLM


# --------------------------------------------------------------------------- safety scan
_FORBIDDEN: list[tuple[str, str]] = [
    (r"\bpremiums?\b", "mentions premiums"),
    (r"\bpric(?:e|es|ed|ing)\b", "mentions pricing"),
    (r"\bcosts?\b|\bcheap\w*|\bdiscounts?\b|\$\s?\d|\bsavings?\b|\bbest rates?\b|\blowest\b", "mentions cost/savings"),
    (r"\bguarant\w+", "makes a guarantee"),
    (r"\beligib\w+|\bqualif(?:y|ies|ied|ication)\b", "mentions eligibility"),
    (r"\bunderwrit\w+|\bapprov(?:e|ed|al|als)\b|\bdecline[ds]?\b", "mentions underwriting/approval"),
    (r"\bbind(?:ing)?\b|\bbound\b", "mentions binding coverage"),
    (r"\b(?:you(?:'ll| will)|we(?:'ll| will| can)) (?:be )?(?:cover|insure|offer|provide|get you|secure)\w*",
     "promises coverage or an outcome"),
    (r"\bfully covered\b|\bcovered for\b|\bwe(?:'ll| will) cover\b", "promises coverage"),
    (r"\bwithin\s+(?:\d+|one|two|three|a|an)\b|\b\d+\s*(?:hours?|business days?|days?)\b", "makes a time commitment"),
    (r"\bsame[- ]day\b|\bright away\b|\bimmediately\b|\bshortly\b|\bas soon as\b|\bbefore the end of\b", "makes a time commitment"),
]
_DATE_RX = re.compile(DATE_PATTERN, re.IGNORECASE)


def _forbidden_hits(text: str) -> list[str]:
    return [label for pattern, label in _FORBIDDEN if re.search(pattern, text, re.IGNORECASE)]


def find_violations(text: str, source_message: str, provided_labels: list[str] = ()) -> list[str]:
    """Return human-readable problems found in a draft (empty list = clean)."""
    problems: list[str] = []
    for pattern, label in _FORBIDDEN:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            problems.append(f"{label}: '{m.group(0)}'")
    src = source_message.lower()
    for m in _DATE_RX.finditer(text):
        if m.group(0).lower() not in src:
            problems.append(f"date/time not written by the customer: '{m.group(0)}'")
    low = text.lower()
    for label in provided_labels:
        core = re.sub(r"\s*\(.*?\)", "", label).strip().lower()   # "Fleet size", "Operating states" ...
        if core and core in low:
            problems.append(f"appears to ask again about something already provided: '{core}'")
    if PLACEHOLDER_SIGNATURE in text:
        problems.append(f"replace the placeholder {PLACEHOLDER_SIGNATURE} with your name before approving")
    return problems


def blocking_violations(problems: list[str]) -> list[str]:
    """Used to vet LLM drafts. Placeholder reminders are advisory; everything else is a violation."""
    return [p for p in problems if "placeholder" not in p]


def approval_blockers(problems: list[str]) -> list[str]:
    """Hard rules checked when a HUMAN approves a draft (backend and UI): no premiums/prices, promises,
    eligibility, time commitments or invented dates. The "may be asking again" heuristic and the signature
    reminder stay advisory, because they can misfire on legitimate human wording."""
    return [p for p in problems if "placeholder" not in p and not p.startswith("appears to ask again")]


# --------------------------------------------------------------------------- composing
# ALL wording that is not a checklist request is fixed text. The LLM never writes any sentence that reaches the
# draft: it may only choose how the missing/unclear fields are grouped (see llm_draft).
INTRO = ("Thank you for reaching out about your commercial trucking insurance. So that our team can review your "
         "submission, could you please send or confirm the following?")
CLOSING = "Once we have these details, a member of our team will review them with you."
NOTHING_MISSING = ("Thank you for sending your details. We have everything we need for a first review, and a "
                   "member of our team will go through your submission.")


def _join(items: list[str]) -> str:
    return "; ".join(items)


def _greeting(ctx: DraftContext) -> str:
    return f"Hi {ctx.customer_first_name}," if ctx.customer_first_name else "Hello,"


def _ask_text(ctx: DraftContext, f: ChecklistField) -> str:
    ask = f.ask
    ev = ctx.evidence.get(f.key, "")
    if ctx.statuses.get(f.key) == "unclear" and ctx.evidence_verified.get(f.key) and ev and len(ev) <= 60 \
            and ev != "not_provided" and not _forbidden_hits(ev):
        ask += f' (you mentioned "{ev}", so please confirm the details)'
    return ask


def _heading(fields: list[ChecklistField]) -> str:
    """Group heading taken from the CONFIGURED field groups (e.g. "Operations", "Fleet and drivers")."""
    names: list[str] = []
    for f in fields:
        if f.group not in names:
            names.append(f.group)
    return " / ".join(names)


def compose(ctx: DraftContext, groups: list[list[ChecklistField]]) -> str:
    """Build the draft from fixed text plus the checklist's own request phrases, one bullet per group."""
    lines = [_greeting(ctx), "", INTRO, ""]
    lines += [f"• {_heading(g)}: {_join([_ask_text(ctx, f) for f in g])}" for g in groups]
    if ctx.deadline_approximate and ctx.deadline_words:
        lines += ["", f'You mentioned your timing as "{ctx.deadline_words}". What is the exact date?']
    lines += ["", CLOSING, "", "Thank you,", PLACEHOLDER_SIGNATURE]
    return "\n".join(lines)


def _config_groups(requested: list[ChecklistField]) -> list[list[ChecklistField]]:
    """The default grouping: by the checklist's configured group."""
    by_group: dict[str, list[ChecklistField]] = {}
    for f in requested:
        by_group.setdefault(f.group, []).append(f)
    return list(by_group.values())


def template_draft(ctx: DraftContext) -> str:
    """Offline, deterministic draft: one bullet per configured group, asks for missing/unclear fields only."""
    requested = ctx.requested()
    if not requested:
        return "\n".join([_greeting(ctx), "", NOTHING_MISSING, "", "Thank you,", PLACEHOLDER_SIGNATURE])
    return compose(ctx, _config_groups(requested))


# --------------------------------------------------------------------------- LLM draft (grouping only)
_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"groups": {"type": "array", "items": {
        "type": "object",
        "properties": {"field_keys": {"type": "array", "items": {"type": "string"}}},
        "required": ["field_keys"]}}},
    "required": ["groups"],
}

_DRAFT_SYSTEM = """You help an insurance broker organise a follow-up request. You are given the list of checklist
fields that are still needed from a customer. Your ONLY job is to GROUP them sensibly.

RULES
- Put related fields together in a few groups (roughly 2 to 4), NOT one group per field.
- Every field_key you are given must appear in exactly one group. Do not add, drop or invent field keys.
- Do NOT write any greeting, sentence, question, heading or other text. The system writes all wording itself.

Return ONE JSON object: {"groups": [{"field_keys": [str]}]}"""


def _llm_user_prompt(ctx: DraftContext) -> str:
    """Only field keys, labels and configured groups are sent: no customer text, so nothing to inject into."""
    still = [{"field_key": f.key, "label": f.label, "configured_group": f.group} for f in ctx.requested()]
    return "STILL NEEDED (JSON):\n" + json.dumps(still, ensure_ascii=False)


def llm_draft(ctx: DraftContext, client: LLMClient) -> str:
    """Ask the LLM only for a GROUPING of the missing/unclear field keys, validate it, and compose the text
    from fixed wording and the checklist's request phrases. Raises LLMCallError / InvalidLLMResponse on any problem."""
    need = ctx.requested()
    if not need:
        raise InvalidLLMResponse("nothing to ask; use the template")
    raw = client.complete_json(_DRAFT_SYSTEM, _llm_user_prompt(ctx), _DRAFT_SCHEMA)
    data = parse_llm_json(raw)
    try:
        key_groups = [list(g["field_keys"]) for g in data["groups"]]
    except (KeyError, TypeError) as exc:
        raise InvalidLLMResponse(f"draft JSON has the wrong shape: {exc}") from exc
    seen = [k for keys in key_groups for k in keys]
    if sorted(seen) != sorted(f.key for f in need):
        raise InvalidLLMResponse("groups do not cover exactly the missing/unclear fields "
                                 f"(asked {sorted(set(seen))}, needed {sorted(f.key for f in need)})")
    if any(not keys for keys in key_groups):
        raise InvalidLLMResponse("draft has an empty group")
    by_key = {f.key: f for f in need}
    # Anything else in the model's JSON (greeting, intro, titles, "text", ...) is ignored by construction.
    return compose(ctx, [[by_key[k] for k in keys] for keys in key_groups])


def generate_draft(ctx: DraftContext, client: Optional[LLMClient]) -> DraftResult:
    """LLM draft if possible and clean; otherwise the template. Never raises."""
    provided_labels = [f.label for f in ctx.provided_fields()]
    if client is None or not client.available:
        return DraftResult(template_draft(ctx), "template", "Offline mode: template draft (no API key).")
    if not ctx.requested():
        return DraftResult(template_draft(ctx), "template", "Nothing is missing, so no questions to draft.")
    try:
        body = llm_draft(ctx, client)
    except (LLMCallError, InvalidLLMResponse) as exc:
        logger.warning("LLM draft failed, using template: %s: %s", type(exc).__name__, exc)
        return DraftResult(template_draft(ctx), "template", f"LLM draft unavailable ({type(exc).__name__}: {exc}); template used.")
    except Exception as exc:                                  # defence in depth: never crash the page
        logger.exception("Unexpected error during LLM draft")
        return DraftResult(template_draft(ctx), "template", f"LLM draft failed ({type(exc).__name__}); template used.")
    bad = blocking_violations(find_violations(body, ctx.message, provided_labels))
    if bad:
        logger.warning("LLM draft rejected by safety scan: %s", bad)
        return DraftResult(template_draft(ctx), "template",
                           "LLM draft was rejected by the safety scan (" + "; ".join(bad) + "); template used.")
    return DraftResult(body, "llm")
