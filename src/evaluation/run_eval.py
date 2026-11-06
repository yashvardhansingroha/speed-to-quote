"""Fill the LLM cache for the evaluation set (the ONLY evaluation code that calls the API).

    python -m src.evaluation.run_eval                 # call the API for every message without a current cache entry
    python -m src.evaluation.run_eval --sleep 4       # pause between calls (useful on free-tier rate limits)
    python -m src.evaluation.run_eval --report        # print the current report (no API calls)

Safe to stop and restart: the cache is saved after every call, and failed calls are retried on the next run.
"""
from __future__ import annotations

import argparse

from src.config import load_checklist
from src.evaluation.cache import refresh_cache
from src.evaluation.export import cache_path, unlabeled_path
from src.evaluation.labels import read_csv
from src.evaluation.report import build_report
from src.extraction.llm_client import LLMClient, load_llm_config


def _fmt(x, pct=True):
    return "n/a" if x is None else (f"{x * 100:.1f}%" if pct else f"{x:.1f}")


def print_report() -> None:
    r = build_report(load_checklist())
    print(f"messages: {r.n_messages} | labels: {r.label_state.message}")
    if r.llm_stats:
        s = r.llm_stats
        print(f"LLM cache ({r.llm_model}): {s.n_llm_path - s.n_missing} of {s.n_llm_path} LLM-path messages current "
              f"({s.n_skipped_short} too short, answered by rules as in the app), {s.n_api_error} API errors, "
              f"{s.n_invalid} invalid, {s.n_valid} valid | invalid rate {_fmt(s.invalid_rate)} | median API-call latency "
              f"{_fmt(r.llm_api_latency_ms, False)} ms | median full-pipeline latency "
              f"{_fmt(r.llm_pipeline_latency_ms, False)} ms")
    for m in (r.rules, r.llm):
        if m:
            print(f"{m.name}: intent acc {_fmt(m.intent_accuracy)} macro-F1 {_fmt(m.intent_macro_f1)} | urgency acc "
                  f"{_fmt(m.urgency_accuracy)} macro-F1 {_fmt(m.urgency_macro_f1)} | missing-field P {_fmt(m.detection.precision)} "
                  f"R {_fmt(m.detection.recall)}")
    for n in r.notes:
        print("note:", n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between API calls")
    ap.add_argument("--report", action="store_true", help="only print the report; make no API calls")
    args = ap.parse_args()
    if args.report:
        print_report()
        return
    client = LLMClient(load_llm_config())
    if not client.available:
        raise SystemExit("No API key configured (GEMINI_API_KEY): nothing to run.")
    checklist = load_checklist()
    _, rows = read_csv(unlabeled_path())
    refresh_cache({r["eval_id"]: r["message"] for r in rows}, client, checklist, cache_path(), sleep_s=args.sleep)
    print_report()


if __name__ == "__main__":
    main()
