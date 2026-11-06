"""Completeness calculation, including not_applicable handling and the zero-denominator case."""
import pytest

from src.scoring.completeness import compute_completeness

KEYS = [f"f{i}" for i in range(12)]


def statuses(**overrides):
    base = {k: "missing" for k in KEYS}
    base.update(overrides)
    return base


def test_all_missing():
    c = compute_completeness(statuses())
    assert (c.provided, c.applicable, c.fraction) == (0, 12, 0.0)


def test_counts_and_text_match_the_spec_format():
    s = statuses(**{f"f{i}": "provided" for i in range(6)}, f6="unclear", f11="not_applicable")
    c = compute_completeness(s)
    assert (c.provided, c.applicable, c.total) == (6, 11, 12)
    assert c.text == "Submission completeness: 6 of 11 applicable fields supplied."
    assert c.unclear == ["f6"] and c.not_applicable == ["f11"] and len(c.missing) == 4


def test_not_applicable_is_removed_from_the_denominator_not_counted_as_provided():
    with_na = compute_completeness(statuses(f0="provided", f1="not_applicable"))
    without = compute_completeness(statuses(f0="provided", f1="missing"))
    assert (with_na.provided, with_na.applicable) == (1, 11)
    assert (without.provided, without.applicable) == (1, 12)
    assert with_na.fraction > without.fraction


def test_unclear_does_not_count_as_supplied():
    assert compute_completeness(statuses(f0="unclear")).provided == 0


def test_everything_provided_is_100_percent():
    assert compute_completeness({k: "provided" for k in KEYS}).fraction == 1.0


def test_nothing_applicable_does_not_divide_by_zero():
    c = compute_completeness({k: "not_applicable" for k in KEYS})
    assert (c.provided, c.applicable, c.fraction) == (0, 0, 0.0)
    assert "0 of 0" in c.text


def test_empty_input():
    assert compute_completeness({}).fraction == 0.0


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError):
        compute_completeness({"a": "kinda"})


def test_completeness_from_a_real_extraction(checklist, now):
    from src.extraction.rules import extract_with_rules
    text = ("This is Dana with Prairie Wind Hauling LLC. We run 8 flatbed trucks in Texas and Oklahoma. "
            "Just got our authority. Our policy expires December 5, need a quote for liability and cargo.")
    ex = extract_with_rules(text, checklist, now)
    c = compute_completeness({k: v.status for k, v in ex.fields.items()})
    assert c.not_applicable == ["loss_runs"] and c.applicable == 11 and c.provided >= 4
