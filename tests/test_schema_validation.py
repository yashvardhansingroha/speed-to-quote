"""Pydantic validation of LLM output, including malformed JSON."""
import json

import pytest

from src.extraction.schema import (
    InvalidLLMResponse, build_response_schema, parse_llm_json, validate_llm_payload,
)
from tests.conftest import llm_payload


def test_valid_payload_is_accepted(checklist):
    ex = validate_llm_payload(llm_payload(checklist), checklist.keys)
    assert ex.intent == "new_quote" and set(ex.fields) == set(checklist.keys)


@pytest.mark.parametrize("raw", ["", "   ", "not json at all", "{broken", "[1, 2, 3]", "null", '"text"'])
def test_malformed_json_is_rejected(raw):
    with pytest.raises(InvalidLLMResponse):
        parse_llm_json(raw)


def test_json_inside_markdown_fence_is_tolerated():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_unknown_intent_is_rejected(checklist):
    with pytest.raises(InvalidLLMResponse, match="intent"):
        validate_llm_payload(llm_payload(checklist, intent="sell_me_a_boat"), checklist.keys)


def test_unknown_status_is_rejected(checklist):
    p = llm_payload(checklist)
    p["fields"]["fleet_size"]["status"] = "maybe"
    with pytest.raises(InvalidLLMResponse):
        validate_llm_payload(p, checklist.keys)


def test_missing_checklist_field_is_rejected(checklist):
    p = llm_payload(checklist)
    del p["fields"]["loss_runs"]
    with pytest.raises(InvalidLLMResponse, match="loss_runs"):
        validate_llm_payload(p, checklist.keys)


def test_missing_top_level_key_is_rejected(checklist):
    p = llm_payload(checklist)
    del p["urgency"]
    with pytest.raises(InvalidLLMResponse):
        validate_llm_payload(p, checklist.keys)


def test_extra_keys_are_ignored_and_extra_fields_dropped(checklist):
    p = llm_payload(checklist)
    p["commentary"] = "ignore me"
    p["fields"]["made_up_field"] = {"status": "provided", "value": "x", "evidence": "x"}
    ex = validate_llm_payload(p, checklist.keys)
    assert "made_up_field" not in ex.fields


def test_enum_values_are_normalised_but_not_guessed(checklist):
    ex = validate_llm_payload(llm_payload(checklist, intent=" New Quote ", urgency="HIGH"), checklist.keys)
    assert ex.intent == "new_quote" and ex.urgency == "high"


def test_numbers_and_lists_in_value_are_converted_to_text(checklist):
    p = llm_payload(checklist)
    p["fields"]["fleet_size"] = {"status": "provided", "value": 12, "evidence": "12 trucks"}
    p["fields"]["operating_states"] = {"status": "provided", "value": ["TX", "OK"], "evidence": "Texas and Oklahoma"}
    ex = validate_llm_payload(p, checklist.keys)
    assert ex.fields["fleet_size"].value == "12"
    assert ex.fields["operating_states"].value == "TX, OK"


def test_null_or_blank_evidence_becomes_not_provided(checklist):
    p = llm_payload(checklist)
    p["fields"]["fleet_size"] = {"status": "missing", "value": "", "evidence": None}
    ex = validate_llm_payload(p, checklist.keys)
    assert ex.fields["fleet_size"].evidence == "not_provided" and ex.fields["fleet_size"].value is None


def test_model_cannot_pre_verify_its_own_evidence(checklist):
    """evidence_verified is code-owned; whatever the model sends is overwritten by verification."""
    from src.extraction.evidence import verify_extraction
    p = llm_payload(checklist)
    p["fields"]["fleet_size"] = {"status": "provided", "value": "9", "evidence": "nine trucks",
                                 "evidence_verified": True}
    ex = verify_extraction(validate_llm_payload(p, checklist.keys), "We have a few trucks")
    assert ex.fields["fleet_size"].evidence_verified is False and ex.fields["fleet_size"].status == "unclear"


def test_response_schema_lists_every_checklist_key_as_required(checklist):
    schema = build_response_schema(checklist.keys)
    assert schema["properties"]["fields"]["required"] == checklist.keys
    json.dumps(schema)                                   # must be JSON-serialisable (sent over HTTP)
    assert "$ref" not in json.dumps(schema)              # providers' structured-output modes dislike $ref
