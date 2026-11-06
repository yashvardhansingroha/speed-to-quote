"""Deadline context for short model quotes, end-to-end evidence relevance, urgency history and the default model."""
from datetime import date, datetime

import pytest

from src.extraction.deadlines import resolve_deadline_in_message
from src.extraction.llm_client import DEFAULT_MODEL
from src.extraction.pipeline import run_extraction
from src.scoring.triage import triage_inquiry
from tests.conftest import as_json, llm_payload
from tests.test_pipeline_fallback import StubClient

RECEIVED = datetime(2026, 11, 2, 9, 0)           # the demo reference date (a Monday)


def run_llm(checklist, text, **overrides):
    """Run the whole pipeline with a stubbed LLM answer, then triage. Returns (extraction, triage)."""
    fields = overrides.pop("fields", {})
    payload = llm_payload(checklist, **overrides)
    payload["fields"].update(fields)
    out = run_extraction(text, checklist, RECEIVED, client=StubClient(answer=as_json(payload)))
    assert out.method == "llm"
    return out.extraction, triage_inquiry(text, out.extraction, RECEIVED, RECEIVED)


# ---- 1. the LLM quotes only the date; the message supplies the context --------------------------
def test_date_only_llm_quote_keeps_context_from_the_message(checklist):
    text = "Hi, we run 5 trucks in Texas. Our policy expires December 5, so we need a quote."
    ex, tri = run_llm(checklist, text, deadline={"status": "provided", "value": "December 5", "evidence": "December 5"})
    assert tri.deadline_date == date(2026, 12, 5) and tri.days_until_deadline == 33
    assert tri.priority.factors[1].points == 18          # 31-60 day tier


def test_date_first_wording_with_a_date_only_llm_quote(checklist):
    text = "We need a new insurance quote. November 6 is our policy expiry date."
    ex, tri = run_llm(checklist, text, deadline={"status": "provided", "value": "November 6", "evidence": "November 6"})
    assert tri.deadline_date == date(2026, 11, 6) and tri.priority.factors[1].points == 40


def test_date_only_quote_for_a_purchase_date_still_gives_no_deadline(checklist):
    text = "We bought 3 trucks on March 3, 2027 and need a quote."
    ex, tri = run_llm(checklist, text, deadline={"status": "provided", "value": "March 3, 2027", "evidence": "March 3, 2027"})
    assert tri.deadline_date is None and tri.priority.factors[1].points == 0


def test_quote_with_line_break_differences_is_still_located(checklist):
    text = "Our policy expires\nDecember 5 and we need a quote."
    assert resolve_deadline_in_message(text, ["policy expires December 5"], RECEIVED)[0] == date(2026, 12, 5)


def test_only_dates_inside_the_quote_are_used():
    text = "Our policy expires December 5. We also need it by Friday."
    assert resolve_deadline_in_message(text, ["December 5"], RECEIVED)[0] == date(2026, 12, 5)
    assert resolve_deadline_in_message(text, ["by Friday"], RECEIVED)[0] == date(2026, 11, 6)


# ---- 2. evidence relevance, end to end through the LLM pipeline ---------------------------------
def test_pipeline_rejects_fleet_99_from_12_trucks_and_99_drivers(checklist):
    text = "We have 12 trucks and 99 drivers and want a quote please"
    ex, _ = run_llm(checklist, text, fields={
        "fleet_size": {"status": "provided", "value": "99", "evidence": "12 trucks and 99 drivers"}})
    assert ex.fields["fleet_size"].status == "unclear" and ex.fields["fleet_size"].value is None


def test_pipeline_rejects_unrelated_quote_for_loss_runs_not_applicable(checklist):
    text = "We have 12 trucks in Texas and want a quote please"
    ex, _ = run_llm(checklist, text, fields={
        "loss_runs": {"status": "not_applicable", "value": None, "evidence": "12 trucks in Texas"}})
    assert ex.fields["loss_runs"].status == "unclear"


# ---- 3. urgency from the LLM path: a past event is not a request ----------------------------------
def test_llm_urgency_backed_only_by_historical_today_is_routed_to_review(checklist):
    text = "We need a quote for our fleet. We bought another truck today."
    ex, tri = run_llm(checklist, text, intent="new_quote", intent_evidence="need a quote",
                      urgency="high", urgency_evidence="bought another truck today")
    assert ex.urgency_evidence_verified is False
    assert tri.needs_review and tri.priority is None


def test_llm_urgency_backed_by_a_real_request_is_scored(checklist):
    text = "We need a quote for our fleet today."
    ex, tri = run_llm(checklist, text, intent="new_quote", intent_evidence="need a quote",
                      urgency="high", urgency_evidence="need a quote for our fleet today")
    assert ex.urgency_evidence_verified is True and not tri.needs_review
    assert tri.priority.factors[0].points == 25


def test_llm_urgency_with_a_non_time_cue_is_not_second_guessed(checklist):
    text = "I called yesterday. It's urgent, we need a quote for 4 trucks."
    ex, tri = run_llm(checklist, text, intent="new_quote", intent_evidence="need a quote",
                      urgency="high", urgency_evidence="It's urgent")
    assert ex.urgency_evidence_verified is True


# ---- 4. model default ------------------------------------------------------------------------------
def test_default_model_is_gemini_3_5_flash_lite():
    assert DEFAULT_MODEL == "gemini-3.5-flash-lite"


def test_env_example_uses_the_same_model():
    from src.config import ROOT
    assert "LLM_MODEL=gemini-3.5-flash-lite" in (ROOT / ".env.example").read_text(encoding="utf-8")


def test_config_default_model_when_nothing_is_set(monkeypatch):
    from src.extraction import llm_client
    for name in ("GEMINI_API_KEY", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "dotenv", None)       # ignore the local .env
    assert llm_client.load_llm_config().model == "gemini-3.5-flash-lite"


def test_a_single_line_break_inside_a_sentence_keeps_the_deadline_context():
    from src.extraction.deadlines import find_deadline
    assert find_deadline("Our policy expires\nDecember 5 and we need a quote.", RECEIVED).date == date(2026, 12, 5)


def test_a_blank_line_does_end_the_context():
    from src.extraction.deadlines import find_deadline
    assert find_deadline("We started the business.\n\nMarch 3 was the day.", RECEIVED) is None


def test_prompt_tells_the_model_to_quote_the_counted_noun(checklist):
    from src.extraction.prompts import build_system_prompt
    assert "never just" in build_system_prompt(checklist)
