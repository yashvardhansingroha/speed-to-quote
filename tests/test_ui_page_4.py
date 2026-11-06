"""UI tests for Page 4 (Model Evaluation)."""
from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.config import ROOT, load_checklist
from src.data_generation.build_database import build_database
from src.evaluation.cache import CacheEntry, load_cache, message_fingerprint, prompt_fingerprint, save_cache
from src.evaluation.export import csv_columns, export_unlabeled
from src.evaluation.labels import read_csv
from src.evaluation.predict import predict_rules

APP_PATH = str((ROOT / "streamlit_app.py").resolve())
REAL_EVAL_DIR = ROOT / "eval"


@pytest.fixture()
def eval_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up an isolated DB and temporary eval folder, blocking all outbound network calls."""
    db_path = tmp_path / "page_4.db"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    build_database(db_path, n=20, seed=5)

    real_unl = REAL_EVAL_DIR / "eval_unlabeled.csv"
    if real_unl.exists():
        shutil.copyfile(real_unl, eval_dir / "eval_unlabeled.csv")
    else:
        export_unlabeled(path=eval_dir / "eval_unlabeled.csv", checklist=load_checklist())

    real_cache = REAL_EVAL_DIR / "llm_cache.json"
    if real_cache.exists():
        shutil.copyfile(real_cache, eval_dir / "llm_cache.json")

    def _forbid_network(*args, **kwargs):
        raise AssertionError("Page 4 must never make a live network call")

    monkeypatch.setattr(requests, "post", _forbid_network)
    monkeypatch.setenv("STQ_DB_PATH", str(db_path))
    monkeypatch.setenv("STQ_EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("STQ_FORCE_OFFLINE", "1")
    return eval_dir


def _page_text(at: AppTest) -> str:
    parts: list[str] = []
    for coll in (at.markdown, at.info, at.warning, at.error, at.caption, at.title, at.subheader):
        for el in coll:
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


def _ensure_complete_temp_cache(eval_dir: Path, checklist) -> None:
    """Ensure the temporary eval folder has a cache covering every unlabeled message.

    Entries added here are stub API errors, so they exercise the fallback path without any real model output.
    """
    cch_path = eval_dir / "llm_cache.json"
    cache = load_cache(cch_path)
    _, rows = read_csv(eval_dir / "eval_unlabeled.csv")
    prompt_sha = prompt_fingerprint(checklist)
    for r in rows:
        eid = r["eval_id"]
        entry = cache.get(eid)
        if entry is not None and entry.prompt_sha == prompt_sha and entry.message_sha == message_fingerprint(r["message"]):
            continue
        cache[eid] = CacheEntry(
            message_sha=message_fingerprint(r["message"]),
            prompt_sha=prompt_sha,
            model=entry.model if entry is not None else "gemini-3.5-flash-lite",
            raw=None,
            error="offline_test_stub",
            latency_ms=120.0,
            created_at="2026-11-02T09:00:00",
        )
    save_cache(cch_path, cache)


def _write_temp_labeled_csv(eval_dir: Path, checklist) -> Path:
    """Create a valid eval_labeled.csv strictly inside the temporary eval_dir with deliberate mismatches."""
    unl_path = eval_dir / "eval_unlabeled.csv"
    _, rows = read_csv(unl_path)
    messages = {r["eval_id"]: r["message"] for r in rows}
    received = {r["eval_id"]: datetime.fromisoformat(r["received_date"] + "T12:00:00") for r in rows}
    rules_preds = predict_rules(messages, received, checklist)

    cols = csv_columns(checklist)
    lab_path = eval_dir / "eval_labeled.csv"
    with open(lab_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for idx, r in enumerate(rows):
            eid = r["eval_id"]
            pred = rules_preds[eid]
            # Flip intent and urgency on the first 7 rows so some failure cases are guaranteed.
            if idx < 7:
                intent = "coverage_question" if pred.intent != "coverage_question" else "new_quote"
                urgency = "high" if pred.urgency != "high" else "low"
            else:
                intent = pred.intent
                urgency = pred.urgency
            row_out = {
                "eval_id": eid,
                "received_date": r["received_date"],
                "message": r["message"],
                "intent": intent,
                "urgency": urgency,
            }
            for k in checklist.keys:
                row_out[f"status_{k}"] = pred.statuses[k]
            writer.writerow(row_out)
    return lab_path


def test_page4_awaiting_labels_shows_no_accuracy_and_never_calls_network(eval_env: Path) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav").set_value("Model Evaluation").run()
    assert not at.exception

    text = _page_text(at)
    assert "SYNTHETIC / DEMONSTRATION DATA" in text
    assert "Awaiting labels" in text
    assert "0 malformed responses only means the output matched the JSON schema" in text
    assert "Pipeline Comparison" not in text
    assert len(at.dataframe) == 0
    assert len(at.get("plotly_chart")) == 0

    for m in at.markdown:
        val = str(m.value)
        if val.startswith('<div class="stq-kpi-card"'):
            assert "\\" not in val


def test_page4_invalid_labels_file_shows_validation_errors(eval_env: Path) -> None:
    bad_path = eval_env / "eval_labeled.csv"
    bad_path.write_text("eval_id,intent,urgency\nE001,not_an_intent,high\n", encoding="utf-8")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav").set_value("Model Evaluation").run()
    assert not at.exception
    assert "Awaiting labels" in _page_text(at)
    assert len(at.error) >= 1
    assert len(at.dataframe) == 0


def test_page4_with_temp_labels_and_cache_shows_comparison_heatmaps_and_failures(eval_env: Path) -> None:
    checklist = load_checklist()
    _ensure_complete_temp_cache(eval_env, checklist)
    lab_path = _write_temp_labeled_csv(eval_env, checklist)
    assert lab_path.parent == eval_env
    assert not (REAL_EVAL_DIR / "eval_labeled.csv").exists()

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.radio(key="nav").set_value("Model Evaluation").run()
    assert not at.exception

    text = _page_text(at)
    assert "Pipeline Comparison (Human-Labelled Set)" in text
    assert "Failure Cases (Heuristic Diagnosis)" in text
    assert "heuristic" in text.lower()

    comp_df = at.dataframe[0].value
    assert list(comp_df.columns) == ["Metric", "LLM pipeline", "Rules baseline"]
    metrics_listed = list(comp_df["Metric"])
    assert "Intent accuracy" in metrics_listed
    assert "Intent macro-F1" in metrics_listed
    assert "Urgency accuracy" in metrics_listed
    assert "Missing-field precision (positive = missing or unclear)" in metrics_listed
    assert "Missing-field recall (positive = missing or unclear)" in metrics_listed

    # Two confusion matrix heatmaps (LLM pipeline and Rules baseline)
    assert len(at.get("plotly_chart")) == 2

    # One comparison table, two per-field tables, then one mismatch table per failure case (5 to 8 cases)
    n_failure_dfs = len(at.dataframe) - 3
    assert 5 <= n_failure_dfs <= 8
