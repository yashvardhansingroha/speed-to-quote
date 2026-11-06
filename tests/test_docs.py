"""The documents quote real, computed numbers (and nothing invented), and the repository stays free of UI code and secrets."""
import re

import pytest

from src.analytics import kpis as K
from src.config import ROOT, load_settings, reference_now
from src.data_generation.build_database import build_database
from src.database import db


@pytest.fixture(scope="module")
def default_kpis(tmp_path_factory):
    path = tmp_path_factory.mktemp("docs") / "default.db"
    build_database(path)                                   # default settings: the same 250 inquiries the memo describes
    conn = db.connect(path)
    frames = K.load_frames(conn)
    lo, hi = K.date_bounds(frames)
    k = K.compute_kpis(frames, reference_now(), load_settings()["mature_cohort_days"], lo, hi)
    conn.close()
    return k


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_founder_memo_numbers_match_the_computed_kpis(default_kpis):
    k, memo = default_kpis, read("docs/founder_memo.md")
    expected = [f"**{k.median_hours:.1f} h**", f"**{k.p90_hours:.1f} h**", f"**{k.responded}**", f"**{k.awaiting}**",
                f"**{k.closed_without_response}**", f"**{k.overdue.numerator} of {k.overdue.denominator}**",
                f"**{k.overdue.pct_text}**", f"**{k.completeness_avg * 100:.1f}%**",
                f"**{k.mature.bound.numerator} of {k.mature.bound.denominator}**", f"**{k.mature.bound.pct_text}**",
                f"**{k.mature.quoted.numerator} of {k.mature.quoted.denominator}**"]
    for token in expected:
        assert token in memo, token


def test_memo_is_labelled_synthetic_and_short():
    memo = read("docs/founder_memo.md")
    assert "SYNTHETIC / DEMONSTRATION DATA" in memo and "independent portfolio project" in memo
    assert len(re.findall(r"\w+", memo)) <= 520


def test_memo_makes_no_revenue_or_causal_claims():
    memo = read("docs/founder_memo.md")
    assert not re.search(r"[$£€]|\brevenue\b|\bprofit\b|\bsavings?\b|\bROI\b", memo, re.IGNORECASE)
    assert "I make no estimate of this" in memo


def test_readme_has_every_required_section_and_the_disclaimers():
    readme = read("README.md")
    for heading in ("## The problem", "## The solution", "## Architecture", "## How the priority score works",
                    "## The checklist and why it looks like this", "## Evaluation", "## Limitations", "## Setup",
                    "## Future work"):
        assert heading in readme, heading
    assert "```mermaid" in readme and "SYNTHETIC / DEMONSTRATION DATA" in readme
    assert "does **not** represent universal" in readme and "ready for human review, not ready for a bindable quote" in readme
    assert "small (80 examples)" in readme and "awaiting labels" in readme.lower()


def test_readme_distinguishes_malformed_output_from_accuracy_and_describes_the_routing():
    readme = read("README.md")
    assert "79 of the 80" in readme and "too short" in readme and "never calls the API" in readme
    assert "says nothing about whether the extracted values are correct" in readme
    assert "API-call time" in readme and "full application path" in readme
    assert "Other fields rely on the verbatim-quote check alone" in readme


def test_the_documented_test_count_is_not_a_placeholder():
    assert "{TESTS}" not in read("README.md")


def test_secrets_are_never_committed_and_the_example_is_only_a_template():
    gi = read(".gitignore")
    assert ".env" in gi.splitlines() and "data/*.db" in gi
    example = read(".env.example")
    assert "GEMINI_API_KEY=\n" in example and "gemini-3.5-flash-lite" in example
    assert not (ROOT / ".env").exists()


def test_the_engine_has_no_user_interface_dependency():
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "streamlit" not in text.lower(), path
    assert "streamlit" not in read("requirements.txt").lower()
