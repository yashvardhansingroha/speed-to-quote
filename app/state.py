"""Session state, secrets bridging, per-thread SQLite connections, and inquiry synchronization."""
from __future__ import annotations

import os
import sqlite3
import threading
import tomllib
from pathlib import Path
from typing import Optional

import streamlit as st

from src.config import ROOT
from src.data_generation.build_database import ensure_database
from src.database import db
from src.database import repository as repo
from src.extraction.llm_client import LLMClient, LLMConfig, load_llm_config

PAGES: tuple[str, ...] = (
    "Inquiry Triage",
    "Submission Review",
    "Operations Overview",
    "Model Evaluation",
)

_SECRET_KEYS: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
)

_DB_INIT_LOCK = threading.Lock()
_INITIALIZED_DBS: set[str] = set()
_THREAD_LOCAL = threading.local()


def copy_secrets_to_env() -> None:
    """Copy LLM settings from secrets.toml into os.environ only when a secrets file exists.

    Touching st.secrets when no secrets.toml file exists causes Streamlit to render an error box
    even inside a try/except block. Reading the TOML file directly avoids that side effect.
    """
    candidates = [
        ROOT / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for key in _SECRET_KEYS:
            val = data.get(key)
            if key not in os.environ and val is not None and str(val).strip():
                os.environ[key] = str(val).strip()


def resolve_db_path() -> Path:
    """Return the active SQLite path, honouring STQ_DB_PATH for test isolation."""
    override = os.environ.get("STQ_DB_PATH", "").strip()
    return Path(override).resolve() if override else db.DEFAULT_DB_PATH.resolve()


def get_connection() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, initializing or migrating the DB once per path."""
    target = resolve_db_path()
    key = str(target)

    with _DB_INIT_LOCK:
        if key not in _INITIALIZED_DBS or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            ensure_database(target)
            _INITIALIZED_DBS.add(key)

    conns: dict[str, sqlite3.Connection] = getattr(_THREAD_LOCAL, "conns", None) or {}
    conn = conns.get(key)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
        except sqlite3.Error:
            conn = None

    if conn is None:
        conn = db.connect_app(target)
        conns[key] = conn
        _THREAD_LOCAL.conns = conns
    return conn


def is_forced_offline() -> bool:
    """True when STQ_FORCE_OFFLINE is set in the environment."""
    return os.environ.get("STQ_FORCE_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def get_llm_client() -> LLMClient:
    """Build an LLMClient reflecting environment config, STQ_FORCE_OFFLINE, and the sidebar toggle."""
    cfg = load_llm_config()
    use_llm_toggle = bool(st.session_state.get("use_llm", True))
    if is_forced_offline() or not use_llm_toggle:
        return LLMClient(
            LLMConfig(
                api_key=None,
                base_url=cfg.base_url,
                model=cfg.model,
                timeout_s=cfg.timeout_s,
            )
        )
    return LLMClient(cfg)


def apply_pending_nav() -> None:
    """Apply a queued page transition before the sidebar navigation radio is instantiated."""
    if "nav" not in st.session_state:
        st.session_state["nav"] = PAGES[0]
    target = st.session_state.pop("nav_target", None)
    if target in PAGES:
        st.session_state["nav"] = target


def request_nav(page: str, inquiry_id: Optional[int] = None) -> None:
    """Queue a page jump (and optional inquiry selection) and trigger a rerun."""
    if inquiry_id is not None:
        st.session_state["current_inquiry_id"] = int(inquiry_id)
    st.session_state["nav_target"] = page
    st.rerun()


def format_inquiry_option(row: sqlite3.Row) -> str:
    """Format an inquiry row for selectboxes: '#12 · 2026-10-01 14:20 · open — first 70 chars'."""
    received = str(row["received_at"]).replace("T", " ")[:16]
    snippet = " ".join(str(row["message"]).split())[:70]
    return f"#{row['id']} · {received} · {row['status']} — {snippet}"


def select_inquiry_widget(
    conn: sqlite3.Connection,
    widget_key: str,
    label: str = "Select an inquiry",
) -> Optional[int]:
    """Render an inquiry selectbox kept in sync with st.session_state['current_inquiry_id']."""
    rows = repo.list_inquiries(conn)
    if not rows:
        return None

    by_id = {int(r["id"]): r for r in rows}
    ids = list(by_id.keys())

    current = st.session_state.get("current_inquiry_id")
    if current not in by_id:
        current = ids[0]
        st.session_state["current_inquiry_id"] = current

    st.session_state[widget_key] = current

    def _on_change() -> None:
        st.session_state["current_inquiry_id"] = st.session_state[widget_key]

    chosen = st.selectbox(
        label,
        options=ids,
        format_func=lambda iid: format_inquiry_option(by_id[iid]),
        key=widget_key,
        on_change=_on_change,
    )
    st.session_state["current_inquiry_id"] = int(chosen)
    return int(chosen)
