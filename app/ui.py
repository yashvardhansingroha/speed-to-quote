"""Shared UI components, escaping helpers, styling, banners, and sidebar."""
from __future__ import annotations

import html
import re
from typing import Optional

import streamlit as st

from app.state import PAGES, is_forced_offline
from src.config import reference_now
from src.extraction.llm_client import LLMClient, load_llm_config

# Characters that can trigger Markdown formatting, links, images, tables, math, or HTML.
_MD_SPECIAL_RX = re.compile(r"([\\`*_{}\[\]()#!|<>~$:&])")


def md(text: Optional[ object]) -> str:
    """Escape untrusted text before passing it to st.markdown, st.caption, warnings, or titles."""
    if text is None:
        return ""
    raw = str(text).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return _MD_SPECIAL_RX.sub(r"\\\1", raw)


def message_box(text: Optional[str], extra_class: str = "") -> None:
    """Render multi-line customer or draft text inside a safe HTML block.

    Blank lines inside an HTML block normally terminate the HTML parser in CommonMark and cause the
    remainder of the text to be interpreted as Markdown. Replacing line breaks with '&#10;' and using
    'white-space: pre-wrap' keeps the entire block on a single source line so nothing escapes.
    """
    escaped = html.escape(text or "", quote=True)
    single_line = escaped.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "&#10;")
    cls = f"stq-message-box {extra_class}".strip()
    st.markdown(f'<div class="{cls}">{single_line}</div>', unsafe_allow_html=True)


def inject_css() -> None:
    """Apply layout spacing and theme-adaptive contrast styles."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.2rem !important;
            padding-bottom: 2.5rem !important;
        }
        .stq-synth-badge {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.22rem 0.65rem;
            border-radius: 4px;
            background-color: rgba(217, 119, 6, 0.16);
            color: #d97706;
            border: 1px solid rgba(217, 119, 6, 0.45);
            margin-bottom: 0.35rem;
        }
        .stq-subtitle {
            font-size: 1.02rem;
            opacity: 0.88;
            margin-top: -0.25rem;
            margin-bottom: 0.9rem;
        }
        .stq-message-box {
            white-space: pre-wrap;
            word-break: break-word;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.92rem;
            line-height: 1.5;
            padding: 0.85rem 1rem;
            border-radius: 6px;
            border: 1px solid rgba(128, 128, 128, 0.35);
            background-color: rgba(128, 128, 128, 0.07);
            margin-bottom: 0.75rem;
        }
        .stq-kpi-card {
            padding: 0.85rem 1rem;
            border-radius: 8px;
            border: 1px solid rgba(128, 128, 128, 0.28);
            background-color: rgba(128, 128, 128, 0.05);
            height: 100%;
        }
        .stq-kpi-label {
            font-size: 0.82rem;
            font-weight: 600;
            opacity: 0.85;
            margin-bottom: 0.2rem;
        }
        .stq-kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }
        .stq-kpi-sub {
            font-size: 0.8rem;
            opacity: 0.82;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render sidebar navigation, LLM toggle, fixed demo clock, and synthetic-data notice."""
    with st.sidebar:
        st.markdown('<div class="stq-synth-badge">SYNTHETIC / DEMONSTRATION DATA</div>', unsafe_allow_html=True)
        st.title("Speed-to-Quote")
        st.caption("AI Intake Copilot for Commercial Trucking Insurance")

        page = st.radio("Navigation", options=list(PAGES), key="nav")

        st.divider()
        if "use_llm" not in st.session_state:
            st.session_state["use_llm"] = True

        env_cfg = load_llm_config()
        forced = is_forced_offline()
        st.toggle(
            "Use the LLM when a key is configured",
            key="use_llm",
            disabled=forced,
            help="When turned off (or when no API key is set), extraction and drafting use the rules baseline.",
        )
        if forced:
            st.caption("Offline mode forced via `STQ_FORCE_OFFLINE`.")
        elif not env_cfg.configured:
            st.caption("No API key configured (`Offline mode: rules-based`).")
        elif st.session_state["use_llm"]:
            st.caption(f"LLM enabled (`{md(env_cfg.model)}`).")
        else:
            st.caption("LLM paused via toggle (`Offline mode: rules-based`).")

        st.divider()
        now = reference_now()
        st.caption(f"**Demo clock (fixed):** {now:%a %d %b %Y %H:%M}")
        st.caption(
            "All records are **SYNTHETIC / DEMONSTRATION DATA** (fictional businesses and contacts). "
            "Human approval is required for every draft; **nothing is ever sent**."
        )
    return str(page)


def render_page_header(title: str, subtitle: str) -> None:
    """Render the mandatory synthetic-data badge, page title, and subtitle."""
    st.markdown('<div class="stq-synth-badge">SYNTHETIC / DEMONSTRATION DATA</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="stq-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def render_mode_banner(client: LLMClient) -> None:
    """Render the offline/LLM status banner required on Pages 1 and 2."""
    if not client.available:
        st.info(
            "**Offline mode: rules-based** — No LLM is in use (no API key configured or LLM toggle is off). "
            "Extraction uses the deterministic rules baseline and follow-up drafts use the fixed template."
        )
    else:
        st.caption(
            f"**Active engine:** LLM (`{md(client.config.model)}`) with deterministic evidence verification "
            "and automatic fallback to the rules-based baseline."
        )


def set_flash(kind: str, message: str, page_key: str = "global") -> None:
    """Store a one-shot status or error banner in session state so it survives a rerun."""
    st.session_state[f"flash_{page_key}"] = (kind, message)


def render_flash(page_key: str = "global") -> None:
    """Pop and display any queued flash message at the top of the page."""
    item = st.session_state.pop(f"flash_{page_key}", None)
    if not item:
        return
    kind, message = item
    safe = md(message)
    if kind == "error":
        st.error(safe)
    elif kind == "warning":
        st.warning(safe)
    elif kind == "success":
        st.success(safe)
    else:
        st.info(safe)
