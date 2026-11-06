-- SYNTHETIC / DEMONSTRATION DATA ONLY. Fictional businesses, placeholder contact details.
-- All timestamps are ISO-8601 text ("YYYY-MM-DDTHH:MM:SS"), naive local time.

CREATE TABLE IF NOT EXISTS inquiries (
    id              INTEGER PRIMARY KEY,
    customer_name   TEXT NOT NULL,                 -- fictional
    contact_email   TEXT NOT NULL,                 -- placeholder (example.com)
    contact_phone   TEXT NOT NULL,                 -- placeholder (555-01xx)
    channel         TEXT NOT NULL,                 -- email | web_form | phone_note | sms
    message         TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',  -- open | closed | bound | lost
    is_synthetic    INTEGER NOT NULL DEFAULT 1
);

-- One row per (inquiry, method). method = 'llm' | 'rules'. The latest row per inquiry is "current".
CREATE TABLE IF NOT EXISTS extractions (
    id                INTEGER PRIMARY KEY,
    inquiry_id        INTEGER NOT NULL REFERENCES inquiries(id),
    method            TEXT NOT NULL,
    model             TEXT,
    intent            TEXT NOT NULL,
    urgency           TEXT NOT NULL,
    intent_evidence   TEXT,
    urgency_evidence  TEXT,
    business_description TEXT,
    deadline_text     TEXT,                        -- the customer's own words
    deadline_date     TEXT,                        -- earliest plausible date, resolved by code (YYYY-MM-DD)
    deadline_latest   TEXT,                        -- latest plausible date (differs for "next month" etc.)
    deadline_approximate INTEGER NOT NULL DEFAULT 0,
    needs_review      INTEGER NOT NULL DEFAULT 0,
    review_reason     TEXT,
    fallback_reason   TEXT,                        -- why the LLM was not used (no key, timeout, malformed ...)
    latency_ms        REAL,
    raw_json          TEXT NOT NULL,               -- the full validated extraction
    created_at        TEXT NOT NULL
);

-- Current checklist state per inquiry. 'source' records whether a value came from the
-- machine ('llm' / 'rules') or from a human correction ('manual').
CREATE TABLE IF NOT EXISTS checklist_status (
    inquiry_id        INTEGER NOT NULL REFERENCES inquiries(id),
    field_key         TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('provided','missing','unclear','not_applicable')),
    value             TEXT,
    evidence          TEXT NOT NULL DEFAULT 'not_provided',
    evidence_verified INTEGER,                     -- 1 = quote found in message, 0 = not found, NULL = n/a
    source            TEXT NOT NULL DEFAULT 'rules',
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (inquiry_id, field_key)
);

-- Event history: inquiry_created -> first_response -> followup_sent / customer_reply
--                -> quote_sent -> bound | lost        (or 'closed' for non-quote inquiries)
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    inquiry_id  INTEGER NOT NULL REFERENCES inquiries(id),
    event_type  TEXT NOT NULL CHECK (event_type IN
                ('inquiry_created','first_response','followup_sent','customer_reply',
                 'quote_sent','bound','lost','closed')),
    occurred_at TEXT NOT NULL,
    detail      TEXT
);

CREATE TABLE IF NOT EXISTS followup_tasks (
    id            INTEGER PRIMARY KEY,
    inquiry_id    INTEGER NOT NULL REFERENCES inquiries(id),
    task_type     TEXT NOT NULL,                   -- missing_info | quote_follow_up
    created_at    TEXT NOT NULL,
    due_at        TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('open','done','cancelled')),
    completed_at  TEXT
);

-- Follow-up drafts. Nothing is ever sent: 'approved' only means a human signed it off.
CREATE TABLE IF NOT EXISTS drafts (
    id            INTEGER PRIMARY KEY,
    inquiry_id    INTEGER NOT NULL REFERENCES inquiries(id),
    original_body TEXT NOT NULL,                   -- exactly what the system generated (never edited)
    body          TEXT NOT NULL,                   -- current text, including human edits
    status        TEXT NOT NULL CHECK (status IN ('draft','approved','rejected')),
    generated_by  TEXT NOT NULL,                   -- llm | template
    note          TEXT,                            -- e.g. why the LLM draft was replaced by the template
    checklist_hash TEXT NOT NULL DEFAULT '',       -- fingerprint of the checklist statuses this draft was written for
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- Append-only record of every human action on a draft (or checklist correction).
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY,
    ts              TEXT NOT NULL,
    inquiry_id      INTEGER NOT NULL REFERENCES inquiries(id),
    action          TEXT NOT NULL,                 -- save_draft | approve | reject | correct_field ...
    original_draft  TEXT,
    edited_draft    TEXT,
    reason          TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_inquiry     ON events(inquiry_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_extractions_inq    ON extractions(inquiry_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_inquiry      ON followup_tasks(inquiry_id, status);
CREATE INDEX IF NOT EXISTS idx_audit_inquiry      ON audit_log(inquiry_id, ts);
