-- MedWeave coordination metadata (SPEC §3.2).
--
-- Coordination and observability only. No images, labels, activations,
-- gradients or patient information are ever stored here (SPEC §3.1).
--
-- Additions to the SPEC §3.2 columns, needed by the dashboard:
--   training_sessions: num_rounds, total_turns, current_peer_id
--   compute_assignments: backbone_version, batches_done, batches_total,
--     first_activation_at, completed_at, loss, accuracy, error
--   training_events.event_type also allows 'failure'

CREATE TABLE IF NOT EXISTS institutions (
    institution_id   TEXT PRIMARY KEY,
    display_name     TEXT NOT NULL,
    client_address   TEXT,
    status           TEXT NOT NULL CHECK (status IN ('online', 'offline', 'training')),
    registered_at    TEXT NOT NULL,
    last_seen_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_sessions (
    session_id              TEXT PRIMARY KEY,
    status                  TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    current_round           INTEGER NOT NULL DEFAULT 0,
    num_rounds              INTEGER NOT NULL,
    total_turns             INTEGER NOT NULL,
    current_institution_id  TEXT REFERENCES institutions (institution_id),
    current_peer_id         TEXT REFERENCES institutions (institution_id),
    created_at              TEXT NOT NULL,
    completed_at            TEXT
);

CREATE TABLE IF NOT EXISTS compute_assignments (
    assignment_id        TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL REFERENCES training_sessions (session_id),
    data_owner_id        TEXT NOT NULL REFERENCES institutions (institution_id),
    compute_peer_id      TEXT NOT NULL REFERENCES institutions (institution_id),
    round_number         INTEGER NOT NULL,
    status               TEXT NOT NULL CHECK (status IN ('pending', 'active', 'completed', 'failed')),
    backbone_version     INTEGER NOT NULL,
    batches_done         INTEGER NOT NULL DEFAULT 0,
    batches_total        INTEGER NOT NULL,
    assigned_at          TEXT NOT NULL,
    first_activation_at  TEXT,
    completed_at         TEXT,
    loss                 REAL,
    accuracy             REAL,
    error                TEXT
);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    checkpoint_id  TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES training_sessions (session_id),
    version        INTEGER NOT NULL,
    storage_path   TEXT NOT NULL,
    created_by     TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES training_sessions (session_id),
    institution_id  TEXT REFERENCES institutions (institution_id),
    event_type      TEXT NOT NULL CHECK (
        event_type IN ('registration', 'assignment', 'training', 'checkpoint', 'completion', 'failure')
    ),
    message         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS noise_configurations (
    config_id    TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES training_sessions (session_id),
    mechanism    TEXT NOT NULL,
    noise_scale  REAL NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assignments_session ON compute_assignments (session_id, assigned_at);
CREATE INDEX IF NOT EXISTS idx_events_session ON training_events (session_id, event_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON model_checkpoints (session_id, version);
