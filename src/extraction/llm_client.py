"""A small, swappable client for any OpenAI-compatible chat-completions API.

Default target is Gemini's OpenAI-compatible endpoint (free tier), but changing LLM_BASE_URL /
LLM_MODEL / the key is all it takes to point it at another provider. We use plain `requests`
(no vendor SDK) so the whole network path is ~60 lines you can read and explain.

Safety properties:
  * The API key is read from the environment (or a local .env file) and is only ever placed in the
    Authorization header. It is never logged, printed, or included in an error message.
  * Every call has a timeout and gets exactly ONE retry (timeouts, connection errors, HTTP 429/5xx,
    or HTTP 400 on the strict-schema mode, in which case the retry uses plain JSON mode).
  * All failures surface as LLMCallError; the caller falls back to the rules baseline.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests

from src.config import ROOT

logger = logging.getLogger("speed_to_quote.llm")

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT = 20.0


class LLMCallError(RuntimeError):
    """The LLM could not be reached or returned an HTTP-level error."""


class LLMNotConfigured(LLMCallError):
    """No API key available -> the app runs in offline (rules-based) mode."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: Optional[str]
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = DEFAULT_TIMEOUT

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def __repr__(self) -> str:                      # never leak the key in logs / tracebacks
        return f"LLMConfig(model={self.model!r}, base_url={self.base_url!r}, key={'set' if self.configured else 'unset'})"


def _setting(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name) or default


def load_llm_config() -> LLMConfig:
    """Build the config from the environment, loading a local .env file first if one exists."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    key = _setting("GEMINI_API_KEY") or _setting("LLM_API_KEY")
    return LLMConfig(
        api_key=key,
        base_url=(_setting("LLM_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/"),
        model=_setting("LLM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        timeout_s=float(_setting("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT)) or DEFAULT_TIMEOUT),
    )


class _Retryable(Exception):
    def __init__(self, message: str, downgrade: bool = False):
        super().__init__(message)
        self.downgrade = downgrade                  # True -> retry in plain JSON mode


class LLMClient:
    def __init__(self, config: LLMConfig,
                 http_post: Optional[Callable[..., Any]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self._post = http_post or requests.post     # injectable so tests never touch the network
        self._sleep = sleep

    @property
    def available(self) -> bool:
        return self.config.configured

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        """Ask for a JSON answer; returns the raw text. Raises LLMCallError on failure."""
        if not self.available:
            raise LLMNotConfigured("no API key configured")

        mode = "json_schema"                        # strict structured output first
        last: Optional[Exception] = None
        for attempt in range(2):                    # first try + exactly one retry
            try:
                return self._call(system, user, schema, mode)
            except _Retryable as exc:
                last = exc
                logger.warning("LLM call failed (attempt %d/2): %s", attempt + 1, exc)
                if exc.downgrade:
                    mode = "json_object"
                if attempt == 0:
                    self._sleep(0.5)
        raise LLMCallError(f"LLM call failed after retry: {last}")

    # -- one HTTP round trip ------------------------------------------------------------
    def _call(self, system: str, user: str, schema: dict[str, Any], mode: str) -> str:
        response_format: dict[str, Any] = (
            {"type": "json_schema", "json_schema": {"name": "inquiry_extraction", "schema": schema}}
            if mode == "json_schema" else {"type": "json_object"})
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": response_format,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        url = f"{self.config.base_url}/chat/completions"
        try:
            resp = self._post(url, json=payload, headers=headers, timeout=self.config.timeout_s)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise _Retryable(f"{type(exc).__name__}") from None

        status = resp.status_code
        if status == 400 and mode == "json_schema":
            raise _Retryable("HTTP 400 on strict schema mode", downgrade=True)
        if status == 429 or status >= 500:
            raise _Retryable(f"HTTP {status}")
        if status != 200:
            raise LLMCallError(f"HTTP {status}")    # 401/403/404 ...: retrying will not help
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMCallError("unexpected response shape from LLM API") from None
