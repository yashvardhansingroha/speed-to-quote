"""The offline rules baseline (also the fallback)."""
import pytest

from src.data_generation.messages import generate_inquiries
from src.extraction.evidence import evidence_in_text
from src.extraction.rules import extract_with_rules

SPEC_EXAMPLE = ("Hi, we run a trucking business with 12 trucks operating in Texas and Oklahoma. "
                "Our insurance expires next month. We're looking for a new quote and want to talk to someone this week.")


@pytest.fixture
def run(checklist, now):
    return lambda text: extract_with_rules(text, checklist, now)


def test_spec_example(run):
    ex = run(SPEC_EXAMPLE)
    assert ex.intent == "renewal_or_replacement"
    assert ex.urgency == "high"                                   # "this week"
    assert ex.fields["fleet_size"].value == "12"
    assert ex.fields["operating_states"].value == "TX, OK"
    assert ex.fields["current_prior_coverage"].status == "provided"
    assert ex.fields["business_legal_name"].status == "missing"   # no name in the message
    assert ex.fields["usdot_mc_number"].status == "missing"
    assert ex.deadline.status == "provided"
    assert ex.business_description.status == "provided"


@pytest.mark.parametrize("text", ["", "   ", "Hi", None])
def test_empty_and_very_short_input_never_crash(run, text):
    ex = run(text)
    assert ex.intent in ("unclear", "new_quote")
    assert all(f.status in ("missing", "unclear") for f in ex.fields.values())


def test_all_checklist_keys_are_always_present(run, checklist):
    assert list(run("random words about nothing").fields) == checklist.keys


@pytest.mark.parametrize("text,value", [
    ("We have twelve trucks", "12"), ("fleet of 7", "7"), ("running 25 power units", "25"),
    ("we operate twenty-five tractors", "25"), ("One rig right now", "1"),
])
def test_fleet_size_forms(run, text, value):
    f = run(text).fields["fleet_size"]
    assert f.status == "provided" and f.value == value


def test_fleet_size_ignores_years_and_other_counts(run):
    assert run("I bought 2019 trucks").fields["fleet_size"].status == "missing"
    assert run("5 years in trucking with 3 drivers").fields["fleet_size"].status == "missing"


def test_vague_fleet_size_is_unclear(run):
    assert run("we have a few trucks").fields["fleet_size"].status == "unclear"


@pytest.mark.parametrize("text,expected", [
    ("We run Texas and Oklahoma", "TX, OK"), ("lanes: TX/OK/AR", "TX, OK, AR"),
    ("mostly TX and OK", "TX, OK"), ("hauling in Ohio, Indiana and Kentucky", "OH, IN, KY"),
])
def test_operating_states(run, text, expected):
    f = run(text).fields["operating_states"]
    assert f.status == "provided" and f.value == expected


def test_ambiguous_state_abbreviations_need_a_neighbour(run):
    assert run("OK thanks, talk to you soon").fields["operating_states"].status == "missing"
    assert run("Call me IN the morning OR tomorrow").fields["operating_states"].status == "missing"


def test_nationwide_is_unclear_not_provided(run):
    assert run("we haul nationwide").fields["operating_states"].status == "unclear"


def test_dot_number(run):
    assert run("our USDOT 1234567").fields["usdot_mc_number"].status == "provided"
    assert run("MC# 845221 here").fields["usdot_mc_number"].status == "provided"
    assert run("we have a DOT number but I don't have it handy").fields["usdot_mc_number"].status == "unclear"


def test_loss_runs_applicability_and_status(run):
    assert run("we just got our authority last week").fields["loss_runs"].status == "not_applicable"
    assert run("loss runs attached").fields["loss_runs"].status == "provided"
    assert run("no claims in the last 3 years").fields["loss_runs"].status == "provided"
    assert run("loss runs available on request").fields["loss_runs"].status == "unclear"
    assert run("we need a quote").fields["loss_runs"].status == "missing"


def test_driver_information_needs_count_and_experience(run):
    assert run("5 drivers, all with 8+ years of experience").fields["driver_information"].status == "provided"
    assert run("we have 5 drivers").fields["driver_information"].status == "unclear"
    assert run("tell me more").fields["driver_information"].status == "missing"


def test_vehicle_schedule_types_only_is_unclear(run):
    assert run("we run dump trucks").fields["vehicle_schedule"].status == "unclear"
    assert run("mostly 2019-2022 Freightliner Cascadias").fields["vehicle_schedule"].status == "provided"


@pytest.mark.parametrize("text,intent", [
    ("I need a certificate of insurance for a shipper", "existing_policy_service"),
    ("Following up on my email, haven't heard back", "follow_up"),
    ("Attached are the documents you asked for", "document_or_info_provided"),
    ("Do I need cargo coverage if I only haul my own equipment?", "coverage_question"),
    ("Our policy expires in 3 weeks, can you shop it around", "renewal_or_replacement"),
    ("I'd like a quote for my new trucking company", "new_quote"),
    ("hello there my friend", "unclear"),
])
def test_intent_rules(run, text, intent):
    assert run(text).intent == intent


def test_multi_intent_picks_one_label_without_crashing(run):
    ex = run("Need a quote for two more trucks and also a certificate of insurance for our current policy.")
    assert ex.intent in ("new_quote", "existing_policy_service")


@pytest.mark.parametrize("text,urgency", [
    ("We need coverage ASAP", "high"), ("No rush, just exploring options", "low"),
    ("Looking for a quote soon", "medium"),
])
def test_urgency_rules(run, text, urgency):
    assert run(text).urgency == urgency


def test_deadline_within_14_days_makes_it_high_urgency(run):
    assert run("Our policy expires in 5 days, need a quote").urgency == "high"


def test_every_rules_evidence_quote_is_found_in_the_message(checklist, now):
    """Property test over 150 generated messages: nothing the baseline returns is invented."""
    for g in generate_inquiries(150, 31337, now, 120):
        ex = extract_with_rules(g.message, checklist, g.received_at)
        for name, f in [("business_description", ex.business_description), ("deadline", ex.deadline),
                        *ex.fields.items()]:
            if f.status != "missing":
                assert f.evidence_verified is True, f"{name}: {f.evidence!r} not in {g.message!r}"
                assert evidence_in_text(f.evidence, g.message)
