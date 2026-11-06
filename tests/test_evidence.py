"""Evidence-quote verification: quotes must really appear in the message."""
from src.extraction.evidence import evidence_in_text, verify_extraction
from src.extraction.schema import validate_llm_payload
from tests.conftest import llm_payload

TEXT = "Hi, we run 12 trucks in Texas and Oklahoma.  Our policy expires next month."


def test_exact_quote_is_found():
    assert evidence_in_text("12 trucks in Texas", TEXT)


def test_only_whitespace_differences_are_tolerated():
    assert evidence_in_text("Texas   and\tOklahoma", TEXT)
    assert evidence_in_text("Our policy expires\nnext month", "Our policy expires   next month")


def test_case_changes_are_not_verbatim():
    assert not evidence_in_text("texas and oklahoma", TEXT)
    assert not evidence_in_text("TEXAS AND OKLAHOMA", TEXT)


def test_changed_punctuation_or_quote_characters_are_not_verbatim():
    assert not evidence_in_text("we’re hiring", "Note: we're hiring")
    assert not evidence_in_text("12 trucks.", "We have 12 trucks and more")


def test_changed_word_is_not_found():
    assert not evidence_in_text("15 trucks in Texas", TEXT)


def test_not_provided_and_empty_are_never_evidence():
    assert not evidence_in_text("not_provided", "not_provided is literally in this text")
    assert not evidence_in_text("", TEXT)
    assert not evidence_in_text("a", TEXT)               # too short to prove anything


def _extraction(checklist, **field_overrides):
    p = llm_payload(checklist)
    p["fields"].update(field_overrides)
    return validate_llm_payload(p, checklist.keys)


def test_verified_quote_keeps_status_and_value(checklist):
    ex = verify_extraction(_extraction(checklist, fleet_size={"status": "provided", "value": "12", "evidence": "12 trucks"}), TEXT)
    f = ex.fields["fleet_size"]
    assert (f.status, f.value, f.evidence_verified) == ("provided", "12", True)


def test_invented_quote_is_downgraded_to_unclear_and_value_dropped(checklist):
    ex = verify_extraction(_extraction(
        checklist, fleet_size={"status": "provided", "value": "25", "evidence": "we operate 25 power units"}), TEXT)
    f = ex.fields["fleet_size"]
    assert f.status == "unclear" and f.value is None and f.evidence_verified is False
    assert "25" in f.note                                 # the human can see what the model claimed


def test_provided_without_any_quote_becomes_unclear(checklist):
    ex = verify_extraction(_extraction(checklist, fleet_size={"status": "provided", "value": "12", "evidence": "not_provided"}), TEXT)
    assert ex.fields["fleet_size"].status == "unclear"


def test_missing_field_is_forced_to_empty(checklist):
    ex = verify_extraction(_extraction(
        checklist, fleet_size={"status": "missing", "value": "7", "evidence": "seven trucks"}), TEXT)
    f = ex.fields["fleet_size"]
    assert f.value is None and f.evidence == "not_provided"


def test_unverifiable_unclear_stays_unclear_but_is_flagged(checklist):
    ex = verify_extraction(_extraction(checklist, fleet_size={"status": "unclear", "value": None, "evidence": "lots of rigs"}), TEXT)
    assert ex.fields["fleet_size"].evidence_verified is False


def test_deadline_and_business_description_are_verified_too(checklist):
    p = llm_payload(checklist)
    p["deadline"] = {"status": "provided", "value": "March 3", "evidence": "expires March 3"}
    p["business_description"] = {"status": "provided", "value": "trucking", "evidence": "we run 12 trucks"}
    ex = verify_extraction(validate_llm_payload(p, checklist.keys), TEXT)
    assert ex.deadline.status == "unclear" and ex.deadline.value is None
    assert ex.business_description.status == "provided"


def test_invented_intent_evidence_makes_intent_unclear(checklist):
    p = llm_payload(checklist, intent_evidence="please quote us immediately")
    ex = verify_extraction(validate_llm_payload(p, checklist.keys), TEXT, checklist)
    assert ex.intent == "unclear" and ex.intent_evidence_verified is False
    assert "new_quote" in ex.intent_note                  # a human can see what the model had claimed
