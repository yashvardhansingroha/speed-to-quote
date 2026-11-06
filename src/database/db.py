"""SQLite connection + schema helpers. Plain `sqlite3`, no ORM, so every query is readable."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

from src.config import ROOT

DEFAULT_DB_PATH = ROOT / "data" / "speed_to_quote.db"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

TABLES = (
    "inquiries", "extractions", "checklist_status", "events",
    "followup_tasks", "drafts", "audit_log",
)


def connect(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and rows addressable by column name."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA_VERSION = 3          # v2: drafts.original_body, extractions ranges; v3: drafts.checklist_hash

# In-place upgrades, applied in order, so a schema change does not erase corrections, drafts or the
# audit log. Keyed by the version they upgrade FROM. A file older than the oldest key is rebuilt.
MIGRATIONS: dict[int, list[str]] = {
    2: ["ALTER TABLE drafts ADD COLUMN checklist_hash TEXT NOT NULL DEFAULT ''"],   # '' = unknown -> treated as outdated
}


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables (idempotent) and stamp the schema version."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def schema_is_current(db_path: Union[str, Path]) -> bool:
    """True if the file exists and was built with the current schema version."""
    if not Path(db_path).exists():
        return False
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


def drop_all(conn: sqlite3.Connection) -> None:
    """Used by the build script to rebuild the demo database from scratch."""
    conn.execute("PRAGMA foreign_keys = OFF")
    for table in TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")


def migrate(db_path: Union[str, Path]) -> bool:
    """Upgrade an existing database in place. True if it is now current, False if it must be rebuilt."""
    if not Path(db_path).exists():
        return False
    conn = sqlite3.connect(str(db_path))
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return True
        if version not in MIGRATIONS:
            return False
        while version < SCHEMA_VERSION:
            for statement in MIGRATIONS[version]:
                conn.execute(statement)
            version += 1
            conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
        return True
    finally:
        conn.close()


def connect_app(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """A connection for ONE thread of the web app.

    A web server handles each visitor on its own thread, and a single sqlite3 connection must not be shared between
    threads doing writes (that produced "bad parameter" / "transaction within a transaction" errors under load).
    So a multi-threaded app should open one connection per thread. WAL lets readers and a writer work at the same time, and the busy
    timeout makes a second writer wait briefly instead of failing immediately.
    """
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:             # e.g. a filesystem that cannot do WAL: keep the default mode
        pass
    return conn
