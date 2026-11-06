"""Config files load, validate, and drive behaviour (no line-specific fields hard-coded in code)."""
import pytest
import yaml

from src.config import load_checklist, load_priority_weights


def test_checklist_has_the_twelve_trucking_fields(checklist):
    assert checklist.line == "trucking"
    assert len(checklist.fields) == 12
    assert len(set(checklist.keys)) == 12
    assert "loss_runs" in checklist.keys and "usdot_mc_number" in checklist.keys
    assert "does not represent universal carrier requirements" in checklist.disclaimer


def test_every_field_has_the_text_needed_by_prompt_and_drafts(checklist):
    for f in checklist.fields:
        assert f.label and f.ask and f.group and f.description


def test_duplicate_checklist_keys_are_rejected(tmp_path):
    bad = yaml.safe_load(open("config/trucking_checklist.yaml", encoding="utf-8"))
    bad["fields"].append(dict(bad["fields"][0]))
    p = tmp_path / "bad_checklist.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_checklist(path=p)


def test_priority_weights_load_and_sum_to_100():
    w = load_priority_weights()
    assert sum(w[k]["max_points"] for k in ("urgency", "deadline", "waiting", "followup")) == 100
    assert all("rationale" in w[k] for k in ("urgency", "deadline", "waiting", "followup"))


def test_priority_weights_that_do_not_sum_to_100_are_rejected(tmp_path):
    w = yaml.safe_load(open("config/priority_weights.yaml", encoding="utf-8"))
    w["urgency"]["max_points"] = 30                      # now totals 105
    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump(w), encoding="utf-8")
    with pytest.raises(ValueError, match="100"):
        load_priority_weights(p)
