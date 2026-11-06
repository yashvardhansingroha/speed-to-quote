"""LLM output cache for the evaluation (eval/llm_cache.json).

Why: the Evaluation page must be reproducible and must NEVER call the API when it loads. API calls happen only
when you run `python -m src.evaluation.run_eval`; their raw answers are stored here and every metric is
recomputed from the cache (parsing, validation and evidence checks are re-run offline each time).

Each entry is keyed by eval_id and remembers the message hash, the model and a fingerprint of the prompt, so a
changed message / model / prompt is detected and flagged as stale instead of silently mixed in.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from src.config import Checklist
from src.extraction.llm_client import LLMCallError, LLMClient
from src.extraction.pipeline import skips_llm
from src.extraction.prompts import build_system_prompt, build_user_prompt
from src.extraction.schema import build_response_schema


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def prompt_fingerprint(checklist: Checklist) -> str:
    """Changes whenever the system prompt or the response schema changes."""
    return _sha(build_system_prompt(checklist) + json.dumps(build_response_schema(checklist.keys), sort_keys=True))


def message_fingerprint(message: str) -> str:
    return _sha(message)


@dataclass
class CacheEntry:
    message_sha: str
    prompt_sha: str
    model: str
    raw: Optional[str]              # the model's text, or None if the API call failed
    error: Optional[str]            # e.g. "LLMCallError: HTTP 429", or None
    latency_ms: float
    created_at: str


def load_cache(path: Path) -> dict[str, CacheEntry]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: CacheEntry(**v) for k, v in data.get("entries", {}).items()}


def save_cache(path: Path, entries: dict[str, CacheEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"note": "Raw LLM answers for the evaluation set. Regenerate with: python -m src.evaluation.run_eval",
               "entries": {k: vars(v) for k, v in sorted(entries.items())}}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)                                    # atomic: an interrupted run cannot corrupt the cache


def entry_is_current(entry: Optional[CacheEntry], message: str, model: str, prompt_sha: str) -> bool:
    return bool(entry and entry.message_sha == message_fingerprint(message)
                and entry.model == model and entry.prompt_sha == prompt_sha)


def refresh_cache(messages: dict[str, str], client: LLMClient, checklist: Checklist, path: Path,
                  retry_errors: bool = True, sleep_s: float = 0.0,
                  log: Callable[[str], None] = print) -> dict[str, CacheEntry]:
    """Call the API for every message that has no current, successful cache entry; save after each call.

    This is the ONLY place in the evaluation code that touches the network.
    """
    entries = load_cache(path)
    pfp, model = prompt_fingerprint(checklist), client.config.model
    system, schema = build_system_prompt(checklist), build_response_schema(checklist.keys)
    skipped = [i for i, m in messages.items() if skips_llm(m)]      # the app never sends these to the LLM either
    todo = [i for i, m in messages.items() if i not in skipped and
            (not entry_is_current(entries.get(i), m, model, pfp) or (retry_errors and entries[i].error))]
    log(f"{len(messages) - len(todo) - len(skipped)} cached, {len(todo)} to call, {len(skipped)} too short for the LLM "
        f"(model {model})")
    for n, eval_id in enumerate(todo, 1):
        start = time.perf_counter()
        raw, error = None, None
        try:
            raw = client.complete_json(system, build_user_prompt(messages[eval_id]), schema)
        except LLMCallError as exc:
            error = f"{type(exc).__name__}: {exc}"
        entries[eval_id] = CacheEntry(
            message_fingerprint(messages[eval_id]), pfp, model, raw, error,
            round((time.perf_counter() - start) * 1000, 1), datetime.now().isoformat(timespec="seconds"))
        save_cache(path, entries)
        log(f"[{n}/{len(todo)}] {eval_id}: {'ERROR ' + error if error else 'ok'} ({entries[eval_id].latency_ms:.0f} ms)")
        if sleep_s and n < len(todo):
            time.sleep(sleep_s)
    return entries
