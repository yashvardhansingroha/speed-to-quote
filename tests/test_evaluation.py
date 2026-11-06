"""Evaluation pipeline: label-free export, label validation, metrics, LLM cache, reports.

IMPORTANT: every test that needs labels creates them inside a temporary folder to exercise the CODE.
Nothing here writes to eval/, and the real eval/eval_labeled.csv (yours) is never read or created by the tests.
"""
import csv
import json
import shutil
from datetime import datetime

import pytest

from src.config import ROOT, load_checklist
from src.evaluation import export, metrics as M
from src.evaluation.cache import (
    CacheEntry, entry_is_current, load_cache, message_fingerprint, prompt_fingerprint, refresh_cache, save_cache)
from src.evaluation.labels import load_labels, read_csv
from src.evaluation.predict import predict_llm, predict_rules
from src.evaluation.report import build_report
from src.extraction.llm_client import LLMCallError, LLMConfig
from src.extraction.schema import INTENTS, STATUSES, URGENCIES
from tests.conftest import as_json, llm_payload

CHECK = load_checklist()
STATUS_COLS = [f"status_{k}" for k in CHECK.keys]


# ---------------------------------------------------------------------------------------------- helpers
@pytest.fixture()
def edir(tmp_path, monkeypatch):
    """A throwaway eval folder with a freshly exported (unlabeled) set. STQ_EVAL_DIR points the code at it."""
    d = tmp_path / "eval"
    d.mkdir()
    monkeypatch.setenv("STQ_EVAL_DIR", str(d))
    export.export_unlabeled(d / "eval_unlabeled.csv")
    return d


def write_labeled(d, n_filled=None, mutate=None):
    """TEST ONLY: copy the unlabeled file and fill labels with an arbitrary cyclic pattern."""
    header, rows = read_csv(d / "eval_unlabeled.csv")
    for i, r in enumerate(rows):
        if n_filled is not None and i >= n_filled:
            continue
        r["intent"], r["urgency"] = INTENTS[i % 7], URGENCIES[i % 3]
        for j, c in enumerate(STATUS_COLS):
            r[c] = STATUSES[(i + j) % 4]
    if mutate:
        mutate(rows)
    with open(d / "eval_labeled.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def messages_of(d):
    return {r["eval_id"]: r["message"] for r in read_csv(d / "eval_unlabeled.csv")[1]}


class FakeClient:
    available = True
    config = LLMConfig(api_key="k", model="fake-model")

    def __init__(self, behaviour=None):
        self.calls, self.behaviour = [], behaviour or (lambda msg: as_json(llm_payload(CHECK)))

    def complete_json(self, system, user, schema):
        self.calls.append(user)
        out = self.behaviour(user)
        if isinstance(out, Exception):
            raise out
        return out


# ---------------------------------------------------------------------------------------------- export
def test_export_has_80_unique_rows_and_empty_label_columns(edir):
    header, rows = read_csv(edir / "eval_unlabeled.csv")
    assert len(rows) == 80 and [r["eval_id"] for r in rows] == [f"E{i:03d}" for i in range(1, 81)]
    assert len({r["message"] for r in rows}) == 80
    label_cols = ["intent", "urgency", *STATUS_COLS]
    assert header == ["eval_id", "received_date", "message", *label_cols]
    assert all(r[c] == "" for r in rows for c in label_cols)                # NEVER pre-filled


def test_export_contains_no_hidden_scenario_information(edir):
    body = (edir / "eval_unlabeled.csv").read_text(encoding="utf-8-sig").split("\n", 1)[1]   # everything after the header
    for token in ("scenario", "new_quote", "renewal_or_replacement", "coverage_question", "existing_policy_service",
                  "document_or_info_provided", "not_applicable"):
        assert token not in body, token                     # label vocabulary never appears in the data rows


def test_export_is_deterministic(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    export.export_unlabeled(a)
    export.export_unlabeled(b)
    assert a.read_bytes() == b.read_bytes()


def test_eval_messages_do_not_overlap_the_operational_data(edir, built_db):
    operational = {r[0] for r in built_db.execute("SELECT message FROM inquiries")}
    from src.data_generation.messages import generate_inquiries
    from src.config import load_settings, reference_now
    s = load_settings()["synthetic"]
    operational |= {g.message for g in generate_inquiries(s["operational_count"], s["operational_seed"], reference_now(), s["history_days"])}
    assert not (set(messages_of(edir).values()) & operational)


def test_the_committed_unlabeled_file_matches_a_fresh_export_and_is_label_free(tmp_path):
    committed = ROOT / "eval" / "eval_unlabeled.csv"
    assert committed.exists()
    fresh = tmp_path / "fresh.csv"
    export.export_unlabeled(fresh)
    assert committed.read_bytes() == fresh.read_bytes()
    _, rows = read_csv(committed)
    assert all(not r[c] for r in rows for c in ["intent", "urgency", *STATUS_COLS])


def test_export_never_writes_the_labeled_file(edir):
    assert not (edir / "eval_labeled.csv").exists()


# ---------------------------------------------------------------------------------------------- label loading
def state(d):
    return load_labels(d / "eval_labeled.csv", d / "eval_unlabeled.csv", CHECK)


def test_no_labeled_file_means_awaiting_labels(edir):
    s = state(edir)
    assert s.state == "no_file" and not s.ready and "Awaiting labels" in s.message


def test_blank_labeled_file_is_incomplete(edir):
    shutil.copy(edir / "eval_unlabeled.csv", edir / "eval_labeled.csv")
    s = state(edir)
    assert s.state == "incomplete" and s.n_complete == 0 and s.n_rows == 80 and "Awaiting labels" in s.message


def test_partially_labeled_file_is_not_ready(edir):
    write_labeled(edir, n_filled=50)
    s = state(edir)
    assert s.state == "incomplete" and (s.n_complete, s.n_rows) == (50, 80) and s.labels == {}


def test_fully_labeled_file_is_ready(edir):
    write_labeled(edir)
    s = state(edir)
    assert s.ready and s.n_complete == 80 and len(s.labels) == 80 and s.labels["E001"]["intent"] == INTENTS[0]


def test_invalid_values_are_reported_not_guessed(edir):
    write_labeled(edir, mutate=lambda rows: rows[2].update(intent="new quote please", urgency="urgent"))
    s = state(edir)
    assert s.state == "invalid" and any("E003" in e and "intent" in e for e in s.errors) and any("urgency" in e for e in s.errors)
    assert s.labels == {} and "Awaiting valid labels" in s.message


def test_case_and_spacing_of_values_are_normalised(edir):
    write_labeled(edir, mutate=lambda rows: rows[0].update(intent=" New-Quote ", status_loss_runs="Not Applicable"))
    s = state(edir)
    assert s.ready and s.labels["E001"]["intent"] == "new_quote" and s.labels["E001"]["status_loss_runs"] == "not_applicable"


def test_missing_columns_wrong_or_duplicate_ids_are_invalid(edir):
    write_labeled(edir)
    header, rows = read_csv(edir / "eval_labeled.csv")
    with open(edir / "eval_labeled.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[h for h in header if h != "urgency"])
        w.writeheader()
        w.writerows([{k: v for k, v in r.items() if k != "urgency"} for r in rows])
    assert state(edir).state == "invalid" and "urgency" in state(edir).errors[0]

    write_labeled(edir, mutate=lambda rows: rows[1].update(eval_id="E001"))
    assert state(edir).state == "invalid"
    write_labeled(edir, mutate=lambda rows: rows[1].update(eval_id="X999"))
    assert any("do not match" in e for e in state(edir).errors)


# ---------------------------------------------------------------------------------------------- metrics (known answers)
def test_accuracy():
    assert M.accuracy(list("aabbc"), list("abbba")) == 0.6 and M.accuracy([], []) is None


def test_per_class_and_macro_f1_known_values():
    t, p = list("aabbc"), list("abbba")
    s = {c.label: c for c in M.per_class(t, p, ["a", "b", "c", "d"])}
    assert (s["a"].precision, s["a"].recall, s["a"].f1) == (0.5, 0.5, 0.5)
    assert s["b"].precision == pytest.approx(2 / 3) and s["b"].recall == 1.0 and s["b"].f1 == pytest.approx(0.8)
    assert s["c"].precision is None and s["c"].recall == 0.0 and s["c"].f1 == 0.0
    assert s["d"].support == 0 and s["d"].predicted == 0
    assert M.macro_f1(t, p, ["a", "b", "c", "d"]) == pytest.approx((0.5 + 0.8 + 0.0) / 3)   # 'd' never occurs: left out


def test_macro_f1_includes_a_class_that_is_only_predicted():
    assert M.macro_f1(list("aa"), list("ab"), ["a", "b"]) == pytest.approx((2 / 3 + 0.0) / 2)


def test_perfect_prediction_scores_one():
    t = list("abcabc")
    assert M.accuracy(t, t) == 1.0 and M.macro_f1(t, t, ["a", "b", "c"]) == 1.0


def test_confusion_matrix_cells():
    cm = M.confusion_matrix(list("aabbc"), list("abbba"), ["a", "b", "c"])
    assert cm.loc["a", "a"] == 1 and cm.loc["a", "b"] == 1 and cm.loc["b", "b"] == 2 and cm.loc["c", "a"] == 1
    assert cm.values.sum() == 5 and cm.index.name == "labelled" and cm.columns.name == "predicted"


def test_missing_field_detection_positive_class_is_missing_or_unclear():
    true = ["missing", "unclear", "provided", "provided", "not_applicable", "missing"]
    pred = ["missing", "provided", "missing", "provided", "provided", "unclear"]
    d = M.detection(true, pred)
    assert (d.tp, d.fp, d.fn, d.tn) == (2, 1, 1, 2)                          # unclear counts as positive; n/a as negative
    assert d.precision == pytest.approx(2 / 3) and d.recall == pytest.approx(2 / 3) and d.n == 6


def test_detection_with_no_positives_gives_none_not_a_crash():
    d = M.detection(["provided"] * 3, ["provided"] * 3)
    assert d.precision is None and d.recall is None and d.tn == 3


# ---------------------------------------------------------------------------------------------- cache
def entry(message, model="m", prompt="p", raw="{}", error=None):
    return CacheEntry(message_fingerprint(message), prompt, model, raw, error, 12.5, "2026-11-06T10:00:00")


def test_cache_round_trip_and_staleness(tmp_path):
    path = tmp_path / "c.json"
    save_cache(path, {"E001": entry("hello world")})
    got = load_cache(path)
    assert got["E001"].latency_ms == 12.5 and json.loads(path.read_text(encoding="utf-8"))["entries"]["E001"]["model"] == "m"
    assert entry_is_current(got["E001"], "hello world", "m", "p")
    assert not entry_is_current(got["E001"], "changed", "m", "p")                # message changed
    assert not entry_is_current(got["E001"], "hello world", "other", "p")        # model changed
    assert not entry_is_current(got["E001"], "hello world", "m", "q")            # prompt changed
    assert not entry_is_current(None, "x", "m", "p")
    assert load_cache(tmp_path / "missing.json") == {}


def test_prompt_fingerprint_changes_with_the_prompt():
    assert prompt_fingerprint(CHECK) == prompt_fingerprint(load_checklist())
    other = load_checklist()
    other.fields[0].description += " (edited)"
    assert prompt_fingerprint(other) != prompt_fingerprint(CHECK)


def test_refresh_calls_only_what_is_missing_stale_or_failed(tmp_path):
    path, msgs = tmp_path / "c.json", {"E1": "this is the first message", "E2": "this is the second message", "E3": "this is the third message"}
    client = FakeClient()
    refresh_cache(msgs, client, CHECK, path, log=lambda s: None)
    assert len(client.calls) == 3
    refresh_cache(msgs, client, CHECK, path, log=lambda s: None)
    assert len(client.calls) == 3                                                # everything cached: no calls
    refresh_cache({**msgs, "E2": "this is the second message CHANGED"}, client, CHECK, path, log=lambda s: None)
    assert len(client.calls) == 4 and "CHANGED" in client.calls[-1]              # only the changed one is re-run


def test_api_failures_are_recorded_and_retried_next_time(tmp_path):
    path, msgs = tmp_path / "c.json", {"E1": "this is the first message", "E2": "this is the second message"}
    flaky = FakeClient(lambda m: LLMCallError("HTTP 429") if "first" in m else as_json(llm_payload(CHECK)))
    refresh_cache(msgs, flaky, CHECK, path, log=lambda s: None)
    got = load_cache(path)
    assert got["E1"].raw is None and "HTTP 429" in got["E1"].error and got["E2"].error is None
    ok = FakeClient()
    refresh_cache(msgs, ok, CHECK, path, log=lambda s: None)
    assert len(ok.calls) == 1 and "first" in ok.calls[0] and load_cache(path)["E1"].error is None


def test_cache_is_saved_after_every_call(tmp_path):
    path = tmp_path / "c.json"
    seen = []
    client = FakeClient(lambda m: seen.append(len(load_cache(path))) or as_json(llm_payload(CHECK)))
    refresh_cache({"E1": "aaa bbb ccc ddd eee fff", "E2": "ggg hhh iii jjj kkk lll", "E3": "mmm nnn ooo ppp qqq rrr"}, client, CHECK, path, log=lambda s: None)
    assert seen == [0, 1, 2]                                                      # each call saw the previous ones on disk


# ---------------------------------------------------------------------------------------------- predictions
def test_llm_predictions_from_cache_classify_valid_invalid_error_and_missing(edir):
    msgs = messages_of(edir)
    ids = list(msgs)[:4]
    sub = {i: msgs[i] for i in ids}
    received = {i: datetime(2026, 10, 1, 12) for i in ids}
    rules = predict_rules(sub, received, CHECK)
    good = as_json(llm_payload(CHECK, intent="coverage_question", urgency="low"))
    pfp = prompt_fingerprint(CHECK)
    cache = {ids[0]: entry(sub[ids[0]], "m", pfp, good), ids[1]: entry(sub[ids[1]], "m", pfp, "not json"),
             ids[2]: entry(sub[ids[2]], "m", pfp, None, "LLMCallError: HTTP 503")}      # ids[3] has no entry
    preds, stats = predict_llm(sub, cache, CHECK, "m", pfp, rules)
    assert preds[ids[0]].source == "llm" and preds[ids[0]].intent == "coverage_question"
    assert preds[ids[1]].source == "llm_fallback" and preds[ids[1]].intent == rules[ids[1]].intent
    assert "invalid LLM response" in preds[ids[1]].note
    assert preds[ids[2]].source == "llm_fallback" and "HTTP 503" in preds[ids[2]].note
    assert ids[3] not in preds
    assert (stats.n_valid, stats.n_invalid, stats.n_api_error, stats.n_missing) == (1, 1, 1, 1)
    assert stats.invalid_rate == 0.5 and not stats.complete and stats.api_latencies_ms == [12.5, 12.5]


def test_invalid_rate_counts_schema_failures_not_api_errors(edir):
    sub = dict(list(messages_of(edir).items())[:3])
    rules = predict_rules(sub, {i: datetime(2026, 10, 1) for i in sub}, CHECK)
    pfp = prompt_fingerprint(CHECK)
    bad_enum = as_json(llm_payload(CHECK, intent="buy_a_boat"))
    cache = {i: entry(m, "m", pfp, r) for (i, m), r in zip(sub.items(), [bad_enum, as_json(llm_payload(CHECK)), None])}
    cache["E00X"] = entry("x", "m", pfp)
    _, stats = predict_llm(sub, {k: v for k, v in cache.items() if k in sub}, CHECK, "m", pfp, rules)
    assert stats.n_invalid == 1 and stats.n_valid == 1 and stats.invalid_rate == 0.5


# ---------------------------------------------------------------------------------------------- the report
def fill_cache(d, behaviour=None, only=None):
    msgs = messages_of(d)
    if only:
        msgs = {k: v for k, v in msgs.items() if k in only}
    refresh_cache(msgs, FakeClient(behaviour), CHECK, d / "llm_cache.json", log=lambda s: None)


def test_report_before_export_says_so(tmp_path):
    r = build_report(CHECK, tmp_path)
    assert not r.exported and r.rules is None and r.llm is None


def test_report_without_labels_shows_awaiting_and_no_accuracy_numbers(edir):
    r = build_report(CHECK)
    assert r.exported and r.n_messages == 80 and not r.label_state.ready
    assert r.rules is None and r.llm is None and r.failures == []
    assert r.rules_latency_ms is not None                                       # latency is measurable without labels


def test_report_without_labels_still_shows_measured_cache_stats(edir):
    fill_cache(edir, lambda m: "garbage" if "renew" in m.lower() else as_json(llm_payload(CHECK)))
    r = build_report(CHECK)
    assert r.llm is None and r.llm_stats.complete and r.llm_stats.n_invalid > 0
    assert r.llm_stats.invalid_rate == pytest.approx(r.llm_stats.n_invalid / r.llm_stats.n_responses)
    assert r.llm_stats.n_llm_path == 79 and r.llm_stats.n_skipped_short == 1       # E015 never goes to the LLM
    assert r.llm_api_latency_ms is not None and r.llm_api_latency_ms >= 0
    assert r.llm_pipeline_latency_ms is not None and r.llm_pipeline_latency_ms >= r.llm_api_latency_ms


def test_report_with_labels_but_no_cache_gives_rules_metrics_only(edir):
    write_labeled(edir)
    r = build_report(CHECK)
    assert r.rules is not None and r.llm is None and r.rules.n == 80
    assert 0 <= r.rules.intent_accuracy <= 1 and r.rules.intent_confusion.values.sum() == 80
    assert len(r.rules.per_field) == 12 and 5 <= len(r.failures) <= 8


def test_report_with_labels_and_cache_compares_both_systems(edir):
    write_labeled(edir)
    fill_cache(edir)
    r = build_report(CHECK)
    assert r.llm is not None and r.llm.n == 80 and r.llm_stats.complete and r.llm_model == "fake-model"
    # the stub model answers new_quote/medium for everything, so its accuracy equals the share of new_quote labels
    _, rows = read_csv(edir / "eval_labeled.csv")
    from src.extraction.pipeline import is_too_short
    from src.extraction.rules import extract_with_rules
    msgs = messages_of(edir)

    def app_answer(row):        # what the APP gives: the stub model, except too-short messages which go to rules
        if is_too_short(msgs[row["eval_id"]]):
            ex = extract_with_rules(msgs[row["eval_id"]], CHECK, datetime(2026, 10, 1, 12))
            return ex.intent, ex.urgency
        return "new_quote", "medium"
    assert r.llm.intent_accuracy == pytest.approx(sum(x["intent"] == app_answer(x)[0] for x in rows) / 80)
    assert r.llm.urgency_accuracy == pytest.approx(sum(x["urgency"] == app_answer(x)[1] for x in rows) / 80)


def test_incomplete_cache_withholds_llm_metrics_and_says_why(edir):
    write_labeled(edir)
    fill_cache(edir, only=set(list(messages_of(edir))[:40]))
    r = build_report(CHECK)
    assert r.llm is None and r.rules is not None and r.llm_stats.n_missing == 40
    assert any("incomplete or out of date" in n for n in r.notes)


def test_changed_message_in_the_labeled_file_is_flagged(edir):
    write_labeled(edir, mutate=lambda rows: rows[4].update(message="SOMETHING ELSE ENTIRELY"))
    r = build_report(CHECK)
    assert r.rules is not None and any("differs from eval_unlabeled.csv" in n for n in r.notes)


def test_failure_cases_have_notes_and_are_between_five_and_ten(edir):
    write_labeled(edir)
    fill_cache(edir)
    f = build_report(CHECK).failures
    assert 5 <= len(f) <= 10
    for c in f:
        assert c.errors and c.diagnosis and c.llm is not None and c.message
    sev = [c.severity for c in f]
    assert sev == sorted(sev, reverse=True)


# ---------------------------------------------------------------------------------------------- never calls the API
def test_building_the_report_never_calls_the_api(edir, monkeypatch):
    import requests
    from src.extraction.llm_client import LLMClient
    write_labeled(edir)
    fill_cache(edir)

    def boom(*a, **k):
        raise AssertionError("the evaluation report must not touch the network")
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(LLMClient, "complete_json", boom)
    assert build_report(CHECK).llm is not None


# ---------------------------------------------------------------------------------------------- app equivalence
SHORT_ID = "E015"                                         # "can someone call me": too short for the LLM path


def varied(msg_user):
    """A deterministic stub model that gives different answers for different messages."""
    n = sum(map(ord, msg_user)) % 7
    return as_json(llm_payload(CHECK, intent=INTENTS[n], urgency=URGENCIES[n % 3]))


def test_evaluation_reproduces_the_apps_extraction_for_every_message(edir):
    """For each of the 80 messages, the evaluation's LLM-pipeline prediction equals what run_extraction (the app's own
    code path) returns, including the short-message routing that skips the LLM."""
    from src.extraction.pipeline import run_extraction
    msgs = messages_of(edir)
    fill_cache(edir, varied)
    r = build_report(CHECK)
    assert r.llm_stats.complete and r.llm_stats.n_skipped_short >= 1
    from src.evaluation.report import predict_llm, predict_rules
    received = {i: datetime(2026, 10, 1, 12) for i in msgs}
    rules = predict_rules(msgs, received, CHECK)
    preds, _ = predict_llm(msgs, load_cache(edir / "llm_cache.json"), CHECK, "fake-model", prompt_fingerprint(CHECK), rules)
    app_client = FakeClient(varied)
    mismatches = []
    for i, text in msgs.items():
        app = run_extraction(text, CHECK, received[i], client=app_client).extraction
        p = preds[i]
        if (p.intent, p.urgency, p.statuses) != (app.intent, app.urgency, {k: v.status for k, v in app.fields.items()}):
            mismatches.append(i)
    assert mismatches == []


def test_the_short_message_is_answered_by_rules_even_when_the_cache_disagrees(edir):
    msgs = messages_of(edir)
    assert SHORT_ID in msgs
    # a (stale-looking) cache entry for the short message that says something different from the rules result
    pfp = prompt_fingerprint(CHECK)
    cache = {SHORT_ID: entry(msgs[SHORT_ID], "m", pfp, as_json(llm_payload(CHECK, intent="coverage_question", urgency="high")))}
    sub = {SHORT_ID: msgs[SHORT_ID]}
    rules = predict_rules(sub, {SHORT_ID: datetime(2026, 10, 1, 12)}, CHECK)
    preds, stats = predict_llm(sub, cache, CHECK, "m", pfp, rules)
    p = preds[SHORT_ID]
    assert p.source == "rules_short_message" and p.intent == rules[SHORT_ID].intent and p.intent != "coverage_question"
    assert (stats.n_skipped_short, stats.n_valid, stats.n_invalid, stats.n_missing, stats.n_llm_path) == (1, 0, 0, 0, 0)
    assert stats.api_latencies_ms == [] and stats.invalid_rate is None


def test_a_short_message_needs_no_cache_entry_and_is_never_counted_missing(edir):
    msgs = messages_of(edir)
    rules = predict_rules(msgs, {i: datetime(2026, 10, 1, 12) for i in msgs}, CHECK)
    _, stats = predict_llm(msgs, {}, CHECK, "m", prompt_fingerprint(CHECK), rules)
    assert stats.n_skipped_short == 1 and stats.n_missing == 79


def test_a_malformed_cached_answer_for_a_short_message_does_not_inflate_the_invalid_rate(edir):
    msgs = messages_of(edir)
    pfp = prompt_fingerprint(CHECK)
    cache = {SHORT_ID: entry(msgs[SHORT_ID], "m", pfp, "garbage")}
    rules = predict_rules({SHORT_ID: msgs[SHORT_ID]}, {SHORT_ID: datetime(2026, 10, 1, 12)}, CHECK)
    _, stats = predict_llm({SHORT_ID: msgs[SHORT_ID]}, cache, CHECK, "m", pfp, rules)
    assert stats.n_invalid == 0 and stats.invalid_rate is None


def test_refresh_never_spends_a_call_on_a_short_message(tmp_path):
    client = FakeClient()
    refresh_cache({"E1": "can someone call me", "E2": "we run 5 trucks in Ohio and need a quote"}, client, CHECK,
                  tmp_path / "c.json", log=lambda s: None)
    assert len(client.calls) == 1 and "5 trucks" in client.calls[0] and list(load_cache(tmp_path / "c.json")) == ["E2"]


def test_api_latency_and_full_pipeline_latency_are_reported_separately(edir):
    msgs = messages_of(edir)
    sub = {k: msgs[k] for k in ("E001", SHORT_ID, "E002")}
    rules = predict_rules(sub, {i: datetime(2026, 10, 1, 12) for i in sub}, CHECK)
    pfp = prompt_fingerprint(CHECK)
    good = as_json(llm_payload(CHECK))
    cache = {"E001": entry(sub["E001"], "m", pfp, good), "E002": entry(sub["E002"], "m", pfp, None, "LLMCallError: HTTP 503")}
    for e in cache.values():
        e.latency_ms = 1000.0
    preds, stats = predict_llm(sub, cache, CHECK, "m", pfp, rules)
    assert stats.api_latencies_ms == [1000.0]                                   # only the call that produced an answer
    assert len(stats.pipeline_latencies_ms) == 3                                # every message has a whole-path time
    assert min(stats.pipeline_latencies_ms) < 50                                # the short one: rules only
    assert sorted(stats.pipeline_latencies_ms)[-1] > 1000.0                     # API time + local work (+ rules on errors)
    assert preds[SHORT_ID].latency_ms == rules[SHORT_ID].latency_ms
