"""Phase 1 smoke tests for the Streamlit skeleton and shared helpers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.state import PAGES, get_connection
from app.ui import md
from src.config import ROOT
from src.data_generation.build_database import build_database

APP_PATH = str((ROOT / "streamlit_app.py").resolve())
BADGE = "SYNTHETIC / DEMONSTRATION DATA"


@pytest.fixture()
def ui_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated temporary database, temporary eval dir, and forced offline mode."""
    db_path = tmp_path / "ui_test.db"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    build_database(db_path, n=30, seed=5)

    monkeypatch.setenv("STQ_DB_PATH", str(db_path))
    monkeypatch.setenv("STQ_EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("STQ_FORCE_OFFLINE", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    return db_path


def _texts(elements) -> str:
    return "\n".join(str(getattr(el, "value", "")) for el in elements)


def _main_text(at: AppTest) -> str:
    """Visible text of the main area only (the sidebar is checked separately)."""
    return "\n".join(
        _texts(c) for c in (at.main.markdown, at.main.info, at.main.warning, at.main.error, at.main.caption, at.main.title)
    )


def test_all_four_pages_load_with_badge_and_offline_banner(ui_env: Path) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    for page in PAGES:
        at.sidebar.radio(key="nav").set_value(page)
        at.run()
        assert not at.exception, f"Exception on page {page}: {at.exception}"
        text = _main_text(at)
        assert BADGE in text, f"badge missing on {page}"
        assert page in _texts(at.main.title)
        if page in ("Inquiry Triage", "Submission Review"):
            assert "Offline mode: rules-based" in text


def test_sidebar_has_badge_and_fixed_demo_clock(ui_env: Path) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    side = _texts(at.sidebar.markdown) + _texts(at.sidebar.caption)
    assert BADGE in side
    assert "Demo clock (fixed)" in side
    assert "nothing is ever sent" in side


def test_sidebar_toggle_forces_offline_even_when_api_key_is_set(
    ui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STQ_FORCE_OFFLINE", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key-never-called")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert "Offline mode: rules-based" not in _main_text(at)

    at.sidebar.toggle(key="use_llm").set_value(False)
    at.run()
    assert not at.exception
    assert "Offline mode: rules-based" in _main_text(at)


def test_md_helper_escapes_markdown_and_html_characters() -> None:
    hostile = (
        "[click](http://evil.example) ![img](http://evil.example/x.png) "
        "<script>alert(1)</script> *bold* _it_ # h1 $x$"
    )
    escaped = md(hostile)
    assert r"\[click\]\(http\://evil.example\)" in escaped
    assert r"\!\[img\]\(http\://evil.example/x.png\)" in escaped
    assert r"\<script\>" in escaped
    assert r"\*bold\*" in escaped
    assert r"\_it\_" in escaped
    assert r"\$x\$" in escaped
    assert "\n" not in md("line1\n\nline2")
    assert md(None) == ""


def test_per_thread_connections_work_concurrently(ui_env: Path) -> None:
    def _worker(_: int) -> int:
        conn = get_connection()
        row = conn.execute("SELECT COUNT(*) AS n FROM inquiries").fetchone()
        return int(row["n"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(8)))
    assert results == [30] * 8
