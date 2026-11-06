"""HTTP behaviour of the LLM client: timeout, exactly one retry, no key leaks. No real network."""
import pytest
import requests

from src.extraction import llm_client
from src.extraction.llm_client import LLMCallError, LLMClient, LLMConfig, LLMNotConfigured, load_llm_config
from tests.conftest import FakeResponse

KEY = "sk-super-secret-test-key"
SCHEMA = {"type": "object"}


class Recorder:
    """Plays back a list of responses/exceptions and records every call."""
    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(*script, key=KEY):
    rec = Recorder(*script)
    return LLMClient(LLMConfig(api_key=key, model="m", base_url="https://example.test/v1", timeout_s=7),
                     http_post=rec, sleep=lambda s: None), rec


def test_not_configured_raises_without_calling_network():
    client, rec = make(key=None)
    with pytest.raises(LLMNotConfigured):
        client.complete_json("s", "u", SCHEMA)
    assert rec.calls == []


def test_success_returns_content_and_sends_timeout_and_auth():
    client, rec = make(FakeResponse(200, content='{"ok": true}'))
    assert client.complete_json("sys", "usr", SCHEMA) == '{"ok": true}'
    call = rec.calls[0]
    assert call["url"] == "https://example.test/v1/chat/completions" and call["timeout"] == 7
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert KEY not in str(call["json"]) and call["json"]["response_format"]["type"] == "json_schema"
    assert call["json"]["messages"][1]["content"] == "usr"


def test_timeout_is_retried_exactly_once():
    client, rec = make(requests.Timeout(), FakeResponse(200, content="{}"))
    assert client.complete_json("s", "u", SCHEMA) == "{}" and len(rec.calls) == 2


def test_two_timeouts_raise_after_exactly_two_attempts():
    client, rec = make(requests.Timeout(), requests.ConnectionError())
    with pytest.raises(LLMCallError):
        client.complete_json("s", "u", SCHEMA)
    assert len(rec.calls) == 2


def test_server_error_then_success():
    client, rec = make(FakeResponse(503), FakeResponse(200, content="{}"))
    assert client.complete_json("s", "u", SCHEMA) == "{}" and len(rec.calls) == 2


def test_rate_limit_twice_gives_up():
    client, rec = make(FakeResponse(429), FakeResponse(429), FakeResponse(200, content="{}"))
    with pytest.raises(LLMCallError):
        client.complete_json("s", "u", SCHEMA)
    assert len(rec.calls) == 2                            # never a third attempt


def test_strict_schema_rejection_retries_in_plain_json_mode():
    client, rec = make(FakeResponse(400), FakeResponse(200, content="{}"))
    client.complete_json("s", "u", SCHEMA)
    assert [c["json"]["response_format"]["type"] for c in rec.calls] == ["json_schema", "json_object"]


def test_auth_error_is_not_retried_and_does_not_leak_the_key():
    client, rec = make(FakeResponse(401))
    with pytest.raises(LLMCallError) as exc:
        client.complete_json("s", "u", SCHEMA)
    assert len(rec.calls) == 1 and KEY not in str(exc.value)


def test_unexpected_response_shape_raises_clean_error():
    client, _ = make(FakeResponse(200, body={"unexpected": 1}))
    with pytest.raises(LLMCallError, match="unexpected"):
        client.complete_json("s", "u", SCHEMA)


def test_config_repr_never_shows_the_key():
    assert KEY not in repr(LLMConfig(api_key=KEY)) and "set" in repr(LLMConfig(api_key=KEY))


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("LLM_MODEL", "my-model")
    cfg = load_llm_config()
    assert cfg.configured and cfg.model == "my-model"


def test_config_without_any_key_is_not_configured(monkeypatch):
    for name in ("GEMINI_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "dotenv", None)     # pretend no .env loader
    assert not load_llm_config().configured
