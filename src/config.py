"""Loading and validating the YAML config files.

Everything line-specific (checklist) or tunable (priority weights, rules) lives in /config.
These loaders turn the YAML into typed objects and fail loudly if a file is inconsistent
(for example, if the priority weights do not add up to 100), so a bad edit is caught at
start-up rather than producing silently wrong scores.
"""
from __future__ import annotations

import copy
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DEFAULT_LINE = "trucking"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


# --------------------------------------------------------------------------- checklist
class FieldRules(BaseModel):
    """How the offline rules baseline finds one checklist field."""
    extractor: str = "regex"                       # regex | states | fleet_size
    patterns: list[str] = Field(default_factory=list)
    unclear_patterns: list[str] = Field(default_factory=list)
    na_patterns: list[str] = Field(default_factory=list)


class ChecklistField(BaseModel):
    key: str
    label: str
    group: str
    ask: str
    description: str
    # Optional sanity check that the extracted value is supported by its quote:
    #   integer     -> the number must appear in the quote (as digits or words)
    #   fleet_count -> the number must be tied to vehicle wording ("12 trucks", "fleet of 12")
    #   states  -> every state in the value must appear in the quote (name or abbreviation)
    #   digits  -> every long digit run in the value (e.g. a DOT number) must appear in the quote
    value_check: Optional[Literal["integer", "fleet_count", "states", "digits"]] = None
    rules: FieldRules = Field(default_factory=FieldRules)


class Checklist(BaseModel):
    line: str
    display_name: str
    disclaimer: str
    macros: dict[str, str] = Field(default_factory=dict)
    fields: list[ChecklistField]

    @property
    def keys(self) -> list[str]:
        return [f.key for f in self.fields]

    def get(self, key: str) -> ChecklistField:
        for f in self.fields:
            if f.key == key:
                return f
        raise KeyError(key)


def load_checklist(line: str = DEFAULT_LINE, path: Optional[Path] = None) -> Checklist:
    """Load config/<line>_checklist.yaml (or an explicit path, used by tests)."""
    path = path or CONFIG_DIR / f"{line}_checklist.yaml"
    checklist = Checklist(**_load_yaml(path))
    keys = checklist.keys
    if len(keys) != len(set(keys)):
        raise ValueError("Checklist field keys must be unique")
    _validate_checklist_regexes(checklist)
    return checklist


def _validate_checklist_regexes(checklist: "Checklist") -> None:
    """Fail at load time, naming the field and pattern, instead of with a cryptic regex error mid-request."""
    import re
    from src.extraction.patterns import expand            # imported here: patterns -> deadlines -> textutils (no cycle)
    for f in checklist.fields:
        for kind in ("patterns", "unclear_patterns", "na_patterns"):
            for pattern in getattr(f.rules, kind):
                try:
                    re.compile(expand(pattern, checklist.macros), re.IGNORECASE)
                except re.error as exc:
                    raise ValueError(f"checklist field '{f.key}', {kind}: invalid regular expression "
                                     f"({exc}): {pattern[:80]}") from exc


# --------------------------------------------------------------------------- priority weights
_WEIGHTS_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def load_priority_weights(path: Optional[Path] = None) -> dict[str, Any]:
    """Load and sanity-check config/priority_weights.yaml.

    The file is parsed once and reused (scoring asks for the weights several times per inquiry, and re-parsing the YAML
    each time made the priority inbox ~30x slower). The cache is keyed by path AND modification time, so editing the
    file takes effect on the next call. Callers get their own copy, so changing it cannot affect anyone else.
    """
    path = Path(path or CONFIG_DIR / "priority_weights.yaml")
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _WEIGHTS_CACHE:
        _WEIGHTS_CACHE[key] = _read_priority_weights(path)
    return copy.deepcopy(_WEIGHTS_CACHE[key])


def _read_priority_weights(path: Path) -> dict[str, Any]:
    w = _load_yaml(path)
    total = sum(w[name]["max_points"] for name in ("urgency", "deadline", "waiting", "followup"))
    if total != 100:
        raise ValueError(f"Priority max_points must add up to 100, got {total}")
    # No individual setting may exceed its factor's maximum.
    for level, pts in w["urgency"]["points"].items():
        if pts > w["urgency"]["max_points"]:
            raise ValueError(f"urgency.{level} exceeds urgency.max_points")
    for tier in w["deadline"]["tiers"]:
        if tier["points"] > w["deadline"]["max_points"]:
            raise ValueError("a deadline tier exceeds deadline.max_points")
    for state, pts in w["followup"]["points"].items():
        if pts > w["followup"]["max_points"]:
            raise ValueError(f"followup.{state} exceeds followup.max_points")
    return w


# --------------------------------------------------------------------------- rules + settings
@lru_cache(maxsize=1)
def load_triage_rules() -> dict[str, Any]:
    rules = _load_yaml(CONFIG_DIR / "triage_rules.yaml")
    _validate_triage_regexes(rules)
    return rules


def _validate_triage_regexes(rules: dict[str, Any]) -> None:
    import re
    patterns: list[tuple[str, str]] = []
    for intent, items in rules.get("intent_rules", {}).items():
        patterns += [(f"intent_rules.{intent}", r["re"]) for r in items]
    u = rules.get("urgency_rules", {})
    for level in ("high", "medium", "low"):
        patterns += [(f"urgency_rules.{level}", p) for p in u.get(level, [])]
    if "context_sensitive" in u:
        patterns.append(("urgency_rules.context_sensitive", u["context_sensitive"]))
    patterns += [("business_description_patterns", p) for p in rules.get("business_description_patterns", [])]
    for where, pattern in patterns:
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"triage_rules.yaml {where}: invalid regular expression ({exc}): {pattern[:80]}") from exc


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "settings.yaml")


def reference_now() -> datetime:
    """The fixed 'now' of the demo. All synthetic data and KPIs are computed relative to it."""
    return datetime.fromisoformat(str(load_settings()["reference_now"]))
