"""The LLM pipeline degrades gracefully: no key, API failure, malformed output -> rules baseline."""
import json
import logging

import pytest

from src.extraction.llm_client import LLMCallError, LLMClient, LLMConfig
from src.extraction.pipeline import is_too_short, run_extraction
from tests.conftest import as_json, llm_payload

MSG = "We run 12 trucks in Texas and Oklahoma and our policy expires next month. Please send a quote."


class StubClient:
    """Duck-typed LLMClient: returns a canned answer or raises."""
    available = True
    config = LLMConfig(api_key="stub-key", model="stub-model")

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, 0

    def complete_json(self, system, user, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


def test_no_api_key_uses_rules_and_reports_offline(checklist, now):
    out = run_extraction(MSG, checklist, now, client=LLMClient(LLMConfig(api_key=None)))
    assert out.method == "rules" and out.offline and out.fallback_reason == "no_api_key"
    assert out.extraction.fields["fleet_size"].value == "12"       # still a useful result


def test_blank_key_counts_as_no_key(checklist, now):
    out = run_extraction(MSG, checklist, now, client=LLMClient(LLMConfig(api_key="   ")))
    assert out.offline and out.fallback_reason == "no_api_key"


def test_no_client_at_all_uses_rules(checklist, now):
    assert run_extraction(MSG, checklist, now, client=None).offline


def test_api_error_falls_back_to_rules(checklist, now):
    out = run_extraction(MSG, checklist, now, client=StubClient(error=LLMCallError("HTTP 503")))
    assert out.offline and "LLMCallError" in out.fallback_reason and "503" in out.fallback_reason


def test_unexpected_exception_still_does_not_crash(checklist, now):
    out = run_extraction(MSG, checklist, now, client=StubClient(error=RuntimeError("boom")))
    assert out.offline and "RuntimeError" in out.fallback_reason


@pytest.mark.parametrize("answer", ["", "not json", "{broken", "[]", '{"intent": "new_quote"}'])
def test_malformed_llm_output_falls_back_and_is_logged(checklist, now, answer, caplog):
    with caplog.at_level(logging.WARNING, logger="speed_to_quote.extraction"):
        out = run_extraction(MSG, checklist, now, client=StubClient(answer=answer))
    assert out.offline and "InvalidLLMResponse" in out.fallback_reason
    assert any("LLM extraction failed" in r.message for r in caplog.records)


def test_valid_llm_response_is_used_and_verified(checklist, now):
    p = llm_payload(checklist, intent="renewal_or_replacement", urgency="medium")
    p["fields"]["fleet_size"] = {"status": "provided", "value": "12", "evidence": "12 trucks"}
    p["fields"]["operating_states"] = {"status": "provided", "value": "TX, OK", "evidence": "Texas and Oklahoma"}
    out = run_extraction(MSG, checklist, now, client=StubClient(answer=as_json(p)))
    assert out.method == "llm" and not out.offline and out.model == "stub-model"
    assert out.extraction.fields["fleet_size"].evidence_verified is True


def test_invented_evidence_from_llm_is_marked_unclear(checklist, now):
    p = llm_payload(checklist)
    p["fields"]["fleet_size"] = {"status": "provided", "value": "40", "evidence": "we operate forty trucks"}
    out = run_extraction(MSG, checklist, now, client=StubClient(answer=as_json(p)))
    f = out.extraction.fields["fleet_size"]
    assert out.method == "llm" and f.status == "unclear" and f.value is None


def test_llm_json_in_code_fence_is_accepted(checklist, now):
    answer = "```json\n" + json.dumps(llm_payload(checklist)) + "\n```"
    assert run_extraction(MSG, checklist, now, client=StubClient(answer=answer)).method == "llm"


@pytest.mark.parametrize("text", ["", "   ", None, "Hi", "need quote"])
def test_empty_or_short_input_never_calls_the_llm(checklist, now, text):
    client = StubClient(answer=as_json(llm_payload(checklist)))
    out = run_extraction(text, checklist, now, client=client)
    assert client.calls == 0 and out.offline and out.fallback_reason == "message_too_short"


def test_is_too_short_thresholds():
    assert is_too_short("") and is_too_short(None) and is_too_short("Hi there")
    assert not is_too_short("We need trucking insurance please")


def test_prompts_contain_every_checklist_key_and_the_untrusted_data_warning(checklist):
    from src.extraction.prompts import build_system_prompt, build_user_prompt
    system = build_system_prompt(checklist)
    assert all(k in system for k in checklist.keys)
    assert "untrusted" in system.lower() and "never invent" in system.lower()
    assert MSG in build_user_prompt(MSG)
