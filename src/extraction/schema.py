"""Pydantic models for extraction results.

Two layers:
  * LLMField / LLMExtraction  - exactly what we ask the LLM to return (the "wire contract").
  * FieldExtraction / InquiryExtraction - the same plus flags that only CODE sets
    (evidence_verified, note). The model can never "self-certify" its own evidence.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

NOT_PROVIDED = "not_provided"

FieldStatus = Literal["provided", "missing", "unclear", "not_applicable"]
Intent = Literal[
    "new_quote", "renewal_or_replacement", "coverage_question",
    "document_or_info_provided", "follow_up", "existing_policy_service", "unclear",
]
Urgency = Literal["high", "medium", "low"]

INTENTS: tuple[str, ...] = (
    "new_quote", "renewal_or_replacement", "coverage_question",
    "document_or_info_provided", "follow_up", "existing_policy_service", "unclear",
)
URGENCIES: tuple[str, ...] = ("high", "medium", "low")
STATUSES: tuple[str, ...] = ("provided", "missing", "unclear", "not_applicable")


class InvalidLLMResponse(ValueError):
    """The LLM answered, but the answer is not usable (bad JSON, wrong shape, wrong enum ...)."""


def _clean_enum(v: Any) -> Any:
    """Be forgiving about 'New Quote' / ' HIGH ' - but not about unknown values."""
    if isinstance(v, str):
        return re.sub(r"[\s-]+", "_", v.strip().lower())
    return v


class LLMField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: FieldStatus
    value: Optional[str] = None
    evidence: str = NOT_PROVIDED

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return _clean_enum(v)

    @field_validator("value", mode="before")
    @classmethod
    def _value_to_text(cls, v: Any) -> Any:
        # Models sometimes return 12 instead of "12", or ["TX","OK"] instead of "TX, OK".
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip() or None                 # "" -> None
        if isinstance(v, (list, tuple)):
            return ", ".join(str(x) for x in v)
        if isinstance(v, (int, float, bool)):
            return str(v)
        raise ValueError("value must be text")

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return NOT_PROVIDED
        if not isinstance(v, str):
            raise ValueError("evidence must be text")
        return v.strip()


class FieldExtraction(LLMField):
    """LLMField + code-only flags."""
    evidence_verified: Optional[bool] = None    # True/False after verification; None = nothing to verify
    note: Optional[str] = None                  # why code changed this field, if it did


class LLMExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: Intent
    urgency: Urgency
    intent_evidence: str = NOT_PROVIDED
    urgency_evidence: str = NOT_PROVIDED
    business_description: LLMField
    deadline: LLMField
    fields: dict[str, LLMField]

    @field_validator("intent", "urgency", mode="before")
    @classmethod
    def _enums(cls, v: Any) -> Any:
        return _clean_enum(v)

    @field_validator("intent_evidence", "urgency_evidence", mode="before")
    @classmethod
    def _ev(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return NOT_PROVIDED
        if not isinstance(v, str):
            raise ValueError("evidence must be text")
        return v.strip()


class InquiryExtraction(LLMExtraction):
    business_description: FieldExtraction
    deadline: FieldExtraction
    fields: dict[str, FieldExtraction]
    intent_evidence_verified: Optional[bool] = None
    urgency_evidence_verified: Optional[bool] = None
    intent_note: Optional[str] = None          # set by code when it overrides the model's intent


# --------------------------------------------------------------------------- parsing helpers
def parse_llm_json(raw: str) -> dict[str, Any]:
    """Turn the model's text into a dict. Tolerates ```json fences, nothing more."""
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidLLMResponse("empty response")
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidLLMResponse(f"not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise InvalidLLMResponse("top-level JSON value must be an object")
    return data


def validate_llm_payload(data: dict[str, Any], required_fields: list[str]) -> InquiryExtraction:
    """Validate a parsed LLM answer against the schema AND the checklist.

    Raises InvalidLLMResponse for anything wrong (wrong types, unknown enum, a checklist key
    missing from `fields`). Unknown extra checklist keys are dropped.
    """
    try:
        llm = LLMExtraction.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise InvalidLLMResponse(f"schema validation failed at '{loc}': {first['msg']}") from exc

    missing = [k for k in required_fields if k not in llm.fields]
    if missing:
        raise InvalidLLMResponse(f"checklist fields missing from response: {', '.join(missing)}")

    payload = llm.model_dump()
    payload["fields"] = {k: payload["fields"][k] for k in required_fields}
    return InquiryExtraction.model_validate(payload)


def build_response_schema(required_fields: list[str]) -> dict[str, Any]:
    """JSON schema sent to the LLM. Built by hand (not from Pydantic) because Gemini/OpenAI
    structured-output modes only accept a simple, $ref-free schema, and because the checklist
    keys come from config and must each be listed as required."""
    field_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": list(STATUSES)},
            "value": {"type": "string"},          # empty string when there is no value
            "evidence": {"type": "string"},
        },
        "required": ["status", "value", "evidence"],
    }
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENTS)},
            "urgency": {"type": "string", "enum": list(URGENCIES)},
            "intent_evidence": {"type": "string"},
            "urgency_evidence": {"type": "string"},
            "business_description": field_schema,
            "deadline": field_schema,
            "fields": {
                "type": "object",
                "properties": {k: field_schema for k in required_fields},
                "required": list(required_fields),
            },
        },
        "required": ["intent", "urgency", "intent_evidence", "urgency_evidence",
                     "business_description", "deadline", "fields"],
    }
