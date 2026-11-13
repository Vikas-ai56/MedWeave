"""SQLite store for coordination metadata, written by the coordinator.

Uses the standard-library `sqlite3` module in WAL mode, so the dashboard can
read while a session is writing. One store is shared by the coordinator's
threads (the session loop and the relay's request handlers), so every write
holds a lock.
"""

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = (Path(__file__).with_name("schema.sql")).read_text()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    """A read-only connection for the dashboard. Raises if the file is missing."""
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


class MetadataStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._lock, self._db:
            self._db.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # --- institutions ----------------------------------------------------------

    def register_institution(self, institution_id: str, client_address: str | None = None) -> None:
        stamp = now()
        self._write(
            """
            INSERT INTO institutions (institution_id, display_name, client_address, status, registered_at, last_seen_at)
            VALUES (?, ?, ?, 'online', ?, ?)
            ON CONFLICT (institution_id) DO UPDATE SET
                client_address = excluded.client_address, status = 'online', last_seen_at = excluded.last_seen_at
            """,
            (institution_id, display_name(institution_id), client_address, stamp, stamp),
        )

    def set_institution_status(self, institution_id: str, status: str, seen: bool = True) -> None:
        if seen:
            self._write(
                "UPDATE institutions SET status = ?, last_seen_at = ? WHERE institution_id = ?",
                (status, now(), institution_id),
            )
        else:
            self._write("UPDATE institutions SET status = ? WHERE institution_id = ?", (status, institution_id))

    # --- sessions --------------------------------------------------------------

    def start_session(self, num_rounds: int, total_turns: int) -> str:
        session_id = new_id()
        self._write(
            """
            INSERT INTO training_sessions (session_id, status, current_round, num_rounds, total_turns, created_at)
            VALUES (?, 'running', 0, ?, ?, ?)
            """,
            (session_id, num_rounds, total_turns, now()),
        )
        return session_id

    def set_current_turn(self, session_id: str, round_number: int, owner_id: str | None, peer_id: str | None) -> None:
        self._write(
            """
            UPDATE training_sessions
            SET current_round = ?, current_institution_id = ?, current_peer_id = ?
            WHERE session_id = ?
            """,
            (round_number, owner_id, peer_id, session_id),
        )

    def finish_session(self, session_id: str, status: str) -> None:
        self._write(
            """
            UPDATE training_sessions
            SET status = ?, completed_at = ?, current_institution_id = NULL, current_peer_id = NULL
            WHERE session_id = ?
            """,
            (status, now(), session_id),
        )

    # --- assignments -----------------------------------------------------------

    def start_assignment(
        self, session_id: str, round_number: int, owner_id: str, peer_id: str, backbone_version: int, batches_total: int
    ) -> str:
        assignment_id = new_id()
        self._write(
            """
            INSERT INTO compute_assignments (
                assignment_id, session_id, data_owner_id, compute_peer_id, round_number,
                status, backbone_version, batches_total, assigned_at
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (assignment_id, session_id, owner_id, peer_id, round_number, backbone_version, batches_total, now()),
        )
        return assignment_id

    def mark_first_activation(self, assignment_id: str) -> None:
        self._write(
            "UPDATE compute_assignments SET first_activation_at = ? WHERE assignment_id = ? AND first_activation_at IS NULL",
            (now(), assignment_id),
        )

    def record_batch_done(self, assignment_id: str) -> None:
        self._write(
            "UPDATE compute_assignments SET batches_done = batches_done + 1 WHERE assignment_id = ?", (assignment_id,)
        )

    def complete_assignment(self, assignment_id: str, loss: float, accuracy: float) -> None:
        self._write(
            """
            UPDATE compute_assignments SET status = 'completed', completed_at = ?, loss = ?, accuracy = ?
            WHERE assignment_id = ?
            """,
            (now(), loss, accuracy, assignment_id),
        )

    def fail_assignment(self, assignment_id: str, error: str) -> None:
        self._write(
            "UPDATE compute_assignments SET status = 'failed', completed_at = ?, error = ? WHERE assignment_id = ?",
            (now(), error, assignment_id),
        )

    # --- checkpoints, events, noise --------------------------------------------

    def add_checkpoint(self, session_id: str, version: int, storage_path: str, created_by: str, created_at: str) -> None:
        self._write(
            """
            INSERT INTO model_checkpoints (checkpoint_id, session_id, version, storage_path, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (new_id(), session_id, version, storage_path, created_by, created_at),
        )

    def add_event(self, session_id: str, institution_id: str | None, event_type: str, message: str) -> None:
        self._write(
            """
            INSERT INTO training_events (session_id, institution_id, event_type, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, institution_id, event_type, message, now()),
        )

    def add_noise_configuration(self, session_id: str, mechanism: str, noise_scale: float) -> None:
        self._write(
            """
            INSERT INTO noise_configurations (config_id, session_id, mechanism, noise_scale, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (new_id(), session_id, mechanism, noise_scale, now()),
        )


def display_name(institution_id: str) -> str:
    """'inst-1' -> 'Institution 1'; other ids are shown as they are."""
    prefix, _, number = institution_id.partition("-")
    return f"Institution {number}" if prefix == "inst" and number.isdigit() else institution_id
