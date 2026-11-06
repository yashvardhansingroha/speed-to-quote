"""The extraction pipeline: try the LLM, validate, verify - or fall back to rules. Never crashes.

                       +--> too short / empty ----------------> rules (LLM skipped)
   message --> choose -+--> no API key -----------------------> rules  ("Offline mode")
                       +--> LLM call --> JSON --> schema check --> evidence check --> result
                                  |         |            |
                                  +---------+------------+--> any failure: log it, use rules
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.config import Checklist, load_settings
from src.extraction.evidence import verify_extraction
from src.extraction.llm_client import LLMCallError, LLMClient
from src.extraction.prompts import build_system_prompt, build_user_prompt
from src.extraction.rules import extract_with_rules
from src.extraction.schema import (
    InquiryExtraction, InvalidLLMResponse, build_response_schema, parse_llm_json,
    validate_llm_payload,
)

logger = logging.getLogger("speed_to_quote.extraction")


class ExtractionOutcome(BaseModel):
    extraction: InquiryExtraction
    method: str                          # "llm" | "rules"
    model: Optional[str] = None
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def offline(self) -> bool:
        """True when the rules baseline produced this result (drives the 'Offline mode' banner)."""
        return self.method == "rules"


def is_too_short(text: Optional[str]) -> bool:
    """Empty or too little text to analyse reliably (thresholds in config/settings.yaml)."""
    s = load_settings()
    t = (text or "").strip()
    return len(t) < s["min_message_chars"] or len(t.split()) < s["min_message_words"]


def is_too_long(text: Optional[str]) -> bool:
    """Longer than the configured cap: not analysed automatically (bounds run time and API cost)."""
    return len(text or "") > load_settings()["max_message_chars"]


def skips_llm(text: Optional[str]) -> bool:
    """Messages the app never sends to the LLM: empty / too short, or too long."""
    return is_too_short(text) or is_too_long(text)


def extract_llm(text: str, checklist: Checklist, client: LLMClient) -> InquiryExtraction:
    """Call the LLM and return a validated, evidence-checked extraction.
    Raises LLMCallError or InvalidLLMResponse on any problem (callers decide what to do)."""
    raw = client.complete_json(
        build_system_prompt(checklist), build_user_prompt(text), build_response_schema(checklist.keys))
    data = parse_llm_json(raw)
    parsed = validate_llm_payload(data, checklist.keys)
    return verify_extraction(parsed, text, checklist)


def run_extraction(text: str, checklist: Checklist, received_at: datetime,
                   client: Optional[LLMClient] = None) -> ExtractionOutcome:
    """Extract from one message using the LLM if possible, otherwise the rules baseline."""
    text = text or ""
    start = time.perf_counter()

    def rules(reason: str, source: Optional[str] = None) -> ExtractionOutcome:
        ex = extract_with_rules(text if source is None else source, checklist, received_at)
        return ExtractionOutcome(extraction=ex, method="rules", fallback_reason=reason,
                                 latency_ms=(time.perf_counter() - start) * 1000)

    if is_too_long(text):
        return rules("message_too_long", source="")           # nothing is analysed; a person reads it
    if is_too_short(text):
        return rules("message_too_short")              # don't spend an API call on "hi"
    if client is None or not client.available:
        return rules("no_api_key")

    try:
        ex = extract_llm(text, checklist, client)
    except (LLMCallError, InvalidLLMResponse) as exc:
        # Log the failure (never the key or the customer text) and degrade gracefully.
        logger.warning("LLM extraction failed, using rules baseline: %s: %s", type(exc).__name__, exc)
        return rules(f"{type(exc).__name__}: {exc}")
    except Exception as exc:                            # defence in depth: the app must not crash
        logger.exception("Unexpected error during LLM extraction")
        return rules(f"unexpected {type(exc).__name__}")

    return ExtractionOutcome(extraction=ex, method="llm", model=client.config.model,
                             latency_ms=(time.perf_counter() - start) * 1000)
