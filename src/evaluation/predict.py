"""Predictions for the evaluation set: the rules baseline (run live, offline) and the LLM pipeline (read from the
cache; this module never calls the API).

"LLM approach" = the pipeline as deployed: the model's answer after schema validation and evidence checks; if the
answer is unusable (API error or invalid response) the app falls back to the rules baseline, so those rows are
scored with the rules result and counted separately in the invalid / error rates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.config import Checklist
from src.evaluation.cache import CacheEntry, entry_is_current
from src.extraction.rules import extract_with_rules
from src.extraction.evidence import verify_extraction
from src.extraction.pipeline import skips_llm
from src.extraction.schema import InquiryExtraction, InvalidLLMResponse, parse_llm_json, validate_llm_payload


@dataclass
class Pred:
    intent: str
    urgency: str
    statuses: dict[str, str]
    source: str                         # "rules" | "llm" | "llm_fallback"
    latency_ms: Optional[float] = None
    note: Optional[str] = None          # e.g. why the LLM result was replaced by the rules result
    extraction: Optional[InquiryExtraction] = field(default=None, repr=False)


@dataclass
class LLMStats:
    n_messages: int = 0
    n_skipped_short: int = 0            # too short for the LLM path: the app answers these with rules, never calling the API
    n_missing: int = 0                  # an LLM call WOULD be made but there is no current cache entry
    n_api_error: int = 0
    n_invalid: int = 0                  # a response arrived but failed JSON / schema validation
    n_valid: int = 0
    api_latencies_ms: list[float] = field(default_factory=list)       # historical API-call time of the cached run
    pipeline_latencies_ms: list[float] = field(default_factory=list)  # whole app path per inquiry (see predict_llm)

    @property
    def n_responses(self) -> int:
        return self.n_valid + self.n_invalid

    @property
    def n_llm_path(self) -> int:
        """Messages the app would actually send to the LLM."""
        return self.n_messages - self.n_skipped_short

    @property
    def invalid_rate(self) -> Optional[float]:
        return self.n_invalid / self.n_responses if self.n_responses else None

    @property
    def complete(self) -> bool:
        return self.n_messages > 0 and self.n_missing == 0


def _pred_from(ex: InquiryExtraction, source: str, latency_ms: Optional[float], note: Optional[str] = None) -> Pred:
    return Pred(ex.intent, ex.urgency, {k: v.status for k, v in ex.fields.items()}, source, latency_ms, note, ex)


def predict_rules(messages: dict[str, str], received: dict[str, datetime], checklist: Checklist) -> dict[str, Pred]:
    out = {}
    for eval_id, text in messages.items():
        start = time.perf_counter()
        ex = extract_with_rules(text, checklist, received[eval_id])
        out[eval_id] = _pred_from(ex, "rules", (time.perf_counter() - start) * 1000)
    return out


def predict_llm(messages: dict[str, str], cache: dict[str, CacheEntry], checklist: Checklist, model: str,
                prompt_sha: str, rules: dict[str, Pred]) -> tuple[dict[str, Pred], LLMStats]:
    """Rebuild what the APP would produce, using cached raw answers (offline; never calls the API).

    Mirrors src.extraction.pipeline.run_extraction:
      * a message that is empty / too short is NOT sent to the LLM: the app uses the rules result;
      * an API error or an invalid answer falls back to the rules result;
      * otherwise the model's answer after schema validation and evidence checks.

    Two latencies are kept apart: the cached API-call time (`api_latencies_ms`, only for calls the app would make),
    and the whole-path time per inquiry (`pipeline_latencies_ms`): API time + the local validation/evidence time
    measured now, plus the rules time whenever rules had to answer.
    """
    stats = LLMStats(n_messages=len(messages))
    out: dict[str, Pred] = {}
    for eval_id, text in messages.items():
        r = rules[eval_id]
        if skips_llm(text):                                      # the app never calls the LLM for these (too short / too long)
            stats.n_skipped_short += 1
            stats.pipeline_latencies_ms.append(r.latency_ms or 0.0)
            out[eval_id] = Pred(r.intent, r.urgency, dict(r.statuses), "rules_short_message", r.latency_ms,
                                "message too short for the LLM path: the app answers with rules", r.extraction)
            continue
        entry = cache.get(eval_id)
        if not entry_is_current(entry, text, model, prompt_sha):
            stats.n_missing += 1
            continue
        if entry.raw is None:                                    # API failure: the app would use rules
            stats.n_api_error += 1
            stats.pipeline_latencies_ms.append(entry.latency_ms + (r.latency_ms or 0.0))
            out[eval_id] = Pred(r.intent, r.urgency, dict(r.statuses), "llm_fallback", None,
                                f"API error: {entry.error}", r.extraction)
            continue
        stats.api_latencies_ms.append(entry.latency_ms)
        start = time.perf_counter()
        try:
            ex = verify_extraction(validate_llm_payload(parse_llm_json(entry.raw), checklist.keys), text, checklist)
        except InvalidLLMResponse as exc:                        # malformed answer: the app falls back to rules
            local_ms = (time.perf_counter() - start) * 1000
            stats.n_invalid += 1
            total = entry.latency_ms + local_ms + (r.latency_ms or 0.0)
            stats.pipeline_latencies_ms.append(total)
            out[eval_id] = Pred(r.intent, r.urgency, dict(r.statuses), "llm_fallback", total,
                                f"invalid LLM response ({exc})", r.extraction)
            continue
        total = entry.latency_ms + (time.perf_counter() - start) * 1000
        stats.n_valid += 1
        stats.pipeline_latencies_ms.append(total)
        out[eval_id] = _pred_from(ex, "llm", total)
    return out, stats
