"""Shared test fixtures and helpers."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.config import load_checklist, reference_now
from src.data_generation.build_database import build_database
from src.database import db


@pytest.fixture(scope="session")
def checklist():
    return load_checklist()


@pytest.fixture(scope="session")
def now() -> datetime:
    return reference_now()


@pytest.fixture(scope="session")
def built_db(tmp_path_factory):
    """A small deterministic synthetic database shared by the whole test session."""
    path = tmp_path_factory.mktemp("db") / "test.db"
    build_database(path, n=90, seed=123)
    conn = db.connect(path)
    yield conn
    conn.close()


def llm_payload(checklist, **overrides) -> dict:
    """A structurally valid LLM answer where every field is 'missing' (tests tweak it)."""
    missing = {"status": "missing", "value": None, "evidence": "not_provided"}
    payload = {
        "intent": "new_quote", "urgency": "medium",
        "intent_evidence": "not_provided", "urgency_evidence": "not_provided",
        "business_description": dict(missing), "deadline": dict(missing),
        "fields": {k: dict(missing) for k in checklist.keys},
    }
    payload.update(overrides)
    return payload


def as_json(payload: dict) -> str:
    return json.dumps(payload)


class FakeResponse:
    """Minimal stand-in for requests.Response."""
    def __init__(self, status_code=200, content=None, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {"choices": [{"message": {"content": content}}]}

    def json(self):
        return self._body
