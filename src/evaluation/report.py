"""Assemble everything the Evaluation page shows. Reads files only: it never calls the LLM API.

Accuracy-type metrics need YOUR labels, so they are computed only when eval/eval_labeled.csv is complete and valid;
otherwise the report says "Awaiting labels". The invalid-response rate and latencies are measurements of the cached
LLM run and do not need labels, so they are shown as soon as the cache exists.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import Checklist
from src.evaluation import metrics as M
from src.evaluation.cache import load_cache, prompt_fingerprint
from src.evaluation.export import cache_path, eval_dir, labeled_path, unlabeled_path
from src.evaluation.failures import FailureCase, select_failures
from src.evaluation.labels import LabelState, load_labels, read_csv
from src.evaluation.predict import LLMStats, Pred, predict_llm, predict_rules
from src.extraction.llm_client import load_llm_config
from src.extraction.schema import INTENTS, URGENCIES
from src.extraction.textutils import collapse_ws


@dataclass
class SystemMetrics:
    name: str
    n: int
    intent_accuracy: Optional[float]
    intent_macro_f1: Optional[float]
    intent_confusion: pd.DataFrame
    urgency_accuracy: Optional[float]
    urgency_macro_f1: Optional[float]
    detection: M.Detection
    per_field: pd.DataFrame
    median_latency_ms: Optional[float]


@dataclass
class EvalReport:
    exported: bool                               # eval_unlabeled.csv exists
    n_messages: int
    label_state: LabelState
    rules: Optional[SystemMetrics] = None
    llm: Optional[SystemMetrics] = None
    llm_stats: Optional[LLMStats] = None
    llm_model: Optional[str] = None
    rules_latency_ms: Optional[float] = None
    llm_api_latency_ms: Optional[float] = None        # cached API-call time (historical)
    llm_pipeline_latency_ms: Optional[float] = None   # whole app path per inquiry, incl. short-message routing
    failures: list[FailureCase] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _system_metrics(name: str, labels: dict[str, dict[str, str]], preds: dict[str, Pred], checklist: Checklist,
                    latencies: list[float]) -> SystemMetrics:
    ids = [i for i in labels if i in preds]
    ti, pi = [labels[i]["intent"] for i in ids], [preds[i].intent for i in ids]
    tu, pu = [labels[i]["urgency"] for i in ids], [preds[i].urgency for i in ids]
    true_s, pred_s, rows = [], [], []
    for key in checklist.keys:
        t = [labels[i][f"status_{key}"] for i in ids]
        p = [preds[i].statuses[key] for i in ids]
        d = M.detection(t, p)
        rows.append({"Field": checklist.get(key).label, "Precision": d.precision, "Recall": d.recall,
                     "Labelled missing/unclear": d.tp + d.fn, "Predicted missing/unclear": d.tp + d.fp})
        true_s += t
        pred_s += p
    return SystemMetrics(
        name, len(ids), M.accuracy(ti, pi), M.macro_f1(ti, pi, INTENTS), M.confusion_matrix(ti, pi, INTENTS),
        M.accuracy(tu, pu), M.macro_f1(tu, pu, URGENCIES), M.detection(true_s, pred_s), pd.DataFrame(rows),
        M.median(latencies))


def build_report(checklist: Checklist, directory: Optional[Path] = None) -> EvalReport:
    unl = unlabeled_path() if directory is None else directory / "eval_unlabeled.csv"
    lab = labeled_path() if directory is None else directory / "eval_labeled.csv"
    cch = cache_path() if directory is None else directory / "llm_cache.json"

    if not unl.exists():
        return EvalReport(False, 0, LabelState("no_file"),
                          notes=["eval/eval_unlabeled.csv has not been exported yet: run python -m src.evaluation.export"])
    _, rows = read_csv(unl)
    messages = {r["eval_id"]: r["message"] for r in rows}
    received = {r["eval_id"]: datetime.fromisoformat(r["received_date"] + "T12:00:00") for r in rows}
    state = load_labels(lab, unl, checklist)
    report = EvalReport(True, len(messages), state)

    if state.ready:   # the labelled copy must contain the same text, otherwise predictions would be judged on other input
        _, lrows = read_csv(lab)
        changed = [r["eval_id"] for r in lrows if collapse_ws(r["message"]) != collapse_ws(messages[r["eval_id"]])]
        if changed:
            report.notes.append(f"The message text differs from eval_unlabeled.csv for {len(changed)} row(s) "
                                f"(e.g. {changed[0]}). Predictions use the text in eval_unlabeled.csv.")

    rules = predict_rules(messages, received, checklist)
    report.rules_latency_ms = M.median([p.latency_ms for p in rules.values()])

    cache = load_cache(cch)
    llm_preds: Optional[dict[str, Pred]] = None
    if cache:
        model = Counter(e.model for e in cache.values()).most_common(1)[0][0] or load_llm_config().model
        llm_preds, stats = predict_llm(messages, cache, checklist, model, prompt_fingerprint(checklist), rules)
        report.llm_stats, report.llm_model = stats, model
        report.llm_api_latency_ms = M.median(stats.api_latencies_ms)
        report.llm_pipeline_latency_ms = M.median(stats.pipeline_latencies_ms)
        if not stats.complete:
            report.notes.append(f"The LLM cache is incomplete or out of date: {stats.n_missing} of {stats.n_messages} "
                                "messages have no current entry. Run python -m src.evaluation.run_eval to fill it.")
            llm_preds = None if stats.n_missing == stats.n_messages else llm_preds

    if state.ready:
        report.rules = _system_metrics("Rules baseline", state.labels, rules, checklist,
                                       [p.latency_ms for p in rules.values() if p.latency_ms is not None])
        if llm_preds is not None and report.llm_stats and report.llm_stats.complete:
            report.llm = _system_metrics("LLM pipeline", state.labels, llm_preds, checklist,
                                         report.llm_stats.pipeline_latencies_ms)
            report.failures = select_failures(messages, state.labels, rules, llm_preds)
        else:
            report.failures = select_failures(messages, state.labels, rules, None)
    return report
