"""Main entry point for the Speed-to-Quote Streamlit application."""
from __future__ import annotations

import streamlit as st

from app.page_eval import render_eval_page
from app.page_ops import render_ops_page
from app.page_review import render_review_page
from app.page_triage import render_triage_page
from app.state import apply_pending_nav, copy_secrets_to_env, get_connection, get_llm_client
from app.ui import inject_css, render_sidebar
from src.config import load_checklist


def main() -> None:
    st.set_page_config(
        page_title="Speed-to-Quote · Trucking Intake Copilot",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    copy_secrets_to_env()
    inject_css()
    apply_pending_nav()

    conn = get_connection()
    checklist = load_checklist()
    page = render_sidebar()
    client = get_llm_client()

    if page == "Inquiry Triage":
        render_triage_page(conn, checklist, client)
    elif page == "Submission Review":
        render_review_page(conn, checklist, client)
    elif page == "Operations Overview":
        render_ops_page(conn, checklist)
    elif page == "Model Evaluation":
        render_eval_page(checklist)


if __name__ == "__main__":
    main()
