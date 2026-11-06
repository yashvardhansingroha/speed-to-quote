"""Synthetic data generator: reproducible, varied, and clearly fake."""
import re

from src.data_generation.build_database import build_database
from src.data_generation.messages import generate_inquiries
from src.database import db


def test_same_seed_gives_identical_output(now):
    a = generate_inquiries(60, 5, now, 120)
    b = generate_inquiries(60, 5, now, 120)
    assert [(g.message, g.received_at) for g in a] == [(g.message, g.received_at) for g in b]


def test_different_seed_gives_different_output(now):
    a = generate_inquiries(60, 5, now, 120)
    b = generate_inquiries(60, 6, now, 120)
    assert [g.message for g in a] != [g.message for g in b]


def test_database_build_is_reproducible(tmp_path):
    p1, p2 = tmp_path / "a.db", tmp_path / "b.db"
    build_database(p1, n=40, seed=9)
    build_database(p2, n=40, seed=9)
    q = "SELECT message, received_at, status FROM inquiries ORDER BY id"
    c1, c2 = db.connect(p1), db.connect(p2)
    assert [tuple(r) for r in c1.execute(q)] == [tuple(r) for r in c2.execute(q)]
    q2 = "SELECT inquiry_id, event_type, occurred_at FROM events ORDER BY id"
    assert [tuple(r) for r in c1.execute(q2)] == [tuple(r) for r in c2.execute(q2)]


def test_messages_are_varied_in_length_and_wording(now):
    gs = generate_inquiries(250, 2026, now, 120)
    msgs = [g.message for g in gs]
    lengths = sorted(len(m) for m in msgs)
    assert len(set(msgs)) > 0.9 * len(msgs)                       # few exact repeats
    assert lengths[0] < 30 and lengths[-1] > 300                  # very short ... long
    assert len({g.scenario_intent for g in gs}) == 7              # every intent appears
    assert len({g.channel for g in gs}) == 4
    assert any(re.search(r"\b(teh|wiht|insurence|polcy)\b", m) for m in msgs)   # typos exist


def test_received_times_are_inside_the_history_window(now):
    from datetime import timedelta
    for g in generate_inquiries(100, 3, now, 120):
        assert now - timedelta(days=120) <= g.received_at < now


def test_contact_details_are_placeholders(now):
    for g in generate_inquiries(100, 3, now, 120):
        assert g.email.endswith("@example.com") and g.phone.startswith("555-01")
