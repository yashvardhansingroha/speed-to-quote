"""Pick 5-10 failure cases and attach a short, HEURISTIC note on why each one failed.

The notes describe mechanisms visible in the data (very short message, two intents present, no keyword matched,
the model's answer was replaced by rules, a quote that does not support a value ...). They are starting points for
your own reading, not a verified root-cause analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.evaluation.metrics import POSITIVE_STATUSES
from src.evaluation.predict import Pred
from src.extraction.pipeline import is_too_short
from src.extraction.rules import intent_scores

MAX_CASES, MIN_CASES = 8, 5


@dataclass
class FailureCase:
    eval_id: str
    message: str
    label: dict[str, str]
    rules: Pred
    llm: Optional[Pred]
    errors: list[str] = field(default_factory=list)       # what is wrong, per system
    diagnosis: list[str] = field(default_factory=list)    # heuristic "why" notes
    severity: int = 0


def _field_errors(label: dict[str, str], pred: Pred) -> list[str]:
    out = []
    for key, status in pred.statuses.items():
        true = label[f"status_{key}"]
        if (true in POSITIVE_STATUSES) != (status in POSITIVE_STATUSES):
            out.append(f"{key}: labelled {true}, predicted {status}")
    return out


def _system_errors(name: str, label: dict[str, str], pred: Pred) -> tuple[list[str], int]:
    errs, sev = [], 0
    if pred.intent != label["intent"]:
        errs.append(f"{name} intent: labelled {label['intent']}, predicted {pred.intent}")
        sev += 3
    if pred.urgency != label["urgency"]:
        errs.append(f"{name} urgency: labelled {label['urgency']}, predicted {pred.urgency}")
        sev += 2
    fe = _field_errors(label, pred)
    errs += [f"{name} field {e}" for e in fe]
    return errs, sev + len(fe)


def _diagnose(message: str, label: dict[str, str], rules: Pred, llm: Optional[Pred]) -> list[str]:
    notes: list[str] = []
    if is_too_short(message):
        notes.append("Very short message: there is little text to classify.")
    scores, hits = intent_scores(message)
    if len(scores) >= 2 and (rules.intent != label["intent"] or (llm and llm.intent != label["intent"])):
        top = sorted(scores, key=scores.get, reverse=True)[:2]
        notes.append(f"Two intents have keyword evidence ({top[0]}, {top[1]}): a multi-intent message.")
    if rules.intent == "unclear" and label["intent"] != "unclear":
        notes.append("No intent keyword matched, so the rules baseline answered 'unclear'.")
    elif rules.intent != label["intent"] and rules.intent in hits:
        notes.append(f"Rules matched “{hits[rules.intent].group(0)}”, which points to {rules.intent}.")
    if llm is not None:
        if llm.source == "rules_short_message":
            notes.append("Too short for the LLM path: the app answers such messages with the rules baseline, so the "
                         "model never sees them.")
        elif llm.source == "llm_fallback":
            notes.append(f"The LLM result was not usable ({llm.note}); the rules baseline answered instead.")
        elif llm.extraction is not None and llm.extraction.intent_note:
            notes.append("The model's supporting quote for intent was not in the message, so intent became 'unclear'.")
        elif llm.extraction is not None and llm.extraction.urgency_evidence_verified is False:
            notes.append("The model's supporting quote for urgency was missing from the message or described a past event.")
        elif llm.intent != label["intent"] and rules.intent == label["intent"]:
            notes.append("The model disagreed with the label where the rules agreed with it: read the message for "
                         "wording the model may have interpreted differently.")
    for src_name, pred in (("rules", rules), ("LLM", llm)):
        if pred is None or pred.extraction is None:
            continue
        for key, status in pred.statuses.items():
            true = label[f"status_{key}"]
            f = pred.extraction.fields.get(key)
            if status == "provided" and true in POSITIVE_STATUSES and f is not None and f.evidence != "not_provided":
                notes.append(f"{src_name} marked {key} provided from the quote “{f.evidence[:50]}”, but it is labelled "
                             f"{true}: check whether the quote really supplies what the checklist needs.")
                break
            if status in POSITIVE_STATUSES and true == "provided":
                notes.append(f"{src_name} did not find {key}, which is labelled provided: its wording may not match "
                             "the patterns or the model's reading.")
                break
    return notes[:4] or ["No single mechanism stands out: compare the message with the labels directly."]


def select_failures(messages: dict[str, str], labels: dict[str, dict[str, str]], rules: dict[str, Pred],
                    llm: Optional[dict[str, Pred]]) -> list[FailureCase]:
    """Most severe failures of the primary system (the LLM pipeline if available, otherwise rules)."""
    cases = []
    for eval_id, label in labels.items():
        errs, sev = _system_errors("rules", label, rules[eval_id])
        if llm is not None and eval_id in llm:
            llm_errs, llm_sev = _system_errors("LLM", label, llm[eval_id])
            errs, sev = errs + llm_errs, llm_sev           # rank by the LLM pipeline's severity
        if not errs:
            continue
        cases.append(FailureCase(eval_id, messages[eval_id], label, rules[eval_id],
                                 llm.get(eval_id) if llm else None, errs, [], sev))
    cases.sort(key=lambda c: (-c.severity, c.eval_id))
    chosen = cases[:MAX_CASES]
    for c in chosen:
        c.diagnosis = _diagnose(c.message, c.label, c.rules, c.llm)
    return chosen
