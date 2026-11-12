"""Builds the dashboard's view of a session from the metadata database.

Read-only. The dashboard shows only coordination metadata: ids, statuses,
versions, counts and training metrics. It never sees images, labels,
activations or gradients (SPEC §4.3 D).
"""

import sqlite3
from datetime import datetime
from typing import Any

STEP_LABELS = [
    ("selected", "Institution selected"),
    ("assigned", "Compute peer assigned"),
    ("activations", "Activations generated"),
    ("noise", "Noise applied"),
    ("backbone", "Backbone computation"),
    ("checkpoint", "Backbone checkpoint updated"),
    ("completed", "Contribution completed"),
]


def list_sessions(db: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = db.execute(
        "SELECT session_id, status, num_rounds, total_turns, created_at, completed_at "
        "FROM training_sessions ORDER BY created_at DESC, rowid DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def build_snapshot(db: sqlite3.Connection, session_id: str | None = None) -> dict[str, Any]:
    session = _session(db, session_id)
    institutions = [dict(row) for row in db.execute("SELECT * FROM institutions ORDER BY institution_id")]
    if session is None:
        return {"session": None, "institutions": _institutions(institutions, None, []), "current_turn": None,
                "turns": [], "model": _model(db, None, []), "noise": None, "events": []}

    sid = session["session_id"]
    assignments = [
        dict(row)
        for row in db.execute(
            "SELECT * FROM compute_assignments WHERE session_id = ? ORDER BY assigned_at, rowid", (sid,)
        )
    ]
    noise = db.execute(
        "SELECT mechanism, noise_scale FROM noise_configurations WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (sid,),
    ).fetchone()
    events = db.execute(
        "SELECT created_at, institution_id, event_type, message FROM training_events "
        "WHERE session_id = ? ORDER BY event_id DESC LIMIT 40",
        (sid,),
    ).fetchall()

    completed = [a for a in assignments if a["status"] == "completed"]
    current = assignments[-1] if assignments else None
    return {
        "session": {
            "id": sid,
            "status": session["status"],
            "current_round": session["current_round"],
            "num_rounds": session["num_rounds"],
            "total_turns": session["total_turns"],
            "completed_turns": len(completed),
            "current_owner": session["current_institution_id"],
            "current_peer": session["current_peer_id"],
            "created_at": session["created_at"],
            "completed_at": session["completed_at"],
        },
        "institutions": _institutions(institutions, session, assignments),
        "current_turn": _current_turn(current, noise is not None),
        "turns": [_turn(a) for a in assignments],
        "model": _model(db, session, completed),
        "noise": dict(noise) if noise else None,
        "events": [
            {"at": e["created_at"], "institution": e["institution_id"], "type": e["event_type"], "message": e["message"]}
            for e in events
        ],
    }


def _session(db: sqlite3.Connection, session_id: str | None) -> sqlite3.Row | None:
    if session_id is not None:
        return db.execute("SELECT * FROM training_sessions WHERE session_id = ?", (session_id,)).fetchone()
    return db.execute("SELECT * FROM training_sessions ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()


def _institutions(rows: list[dict], session: sqlite3.Row | None, assignments: list[dict]) -> list[dict[str, Any]]:
    running = session is not None and session["status"] == "running"
    current = assignments[-1] if assignments else None
    result = []
    for row in rows:
        iid = row["institution_id"]
        role = "idle"
        if running and iid == session["current_institution_id"]:
            role = "owner"
        elif running and iid == session["current_peer_id"]:
            role = "peer"
        own = [a for a in assignments if a["data_owner_id"] == iid and a["status"] == "completed"]
        progress = None
        if role == "owner" and current is not None and current["data_owner_id"] == iid:
            progress = {"done": current["batches_done"], "total": current["batches_total"]}
        result.append(
            {
                "id": iid,
                "name": row["display_name"],
                "status": row["status"],
                "role": role,
                "last_seen_at": row["last_seen_at"],
                "progress": progress,
                "contributions": len(own),
                "computations": sum(1 for a in assignments if a["compute_peer_id"] == iid and a["status"] == "completed"),
                "last_contribution": _turn(own[-1]) if own else None,
            }
        )
    return result


def _current_turn(assignment: dict | None, noise_configured: bool) -> dict[str, Any] | None:
    if assignment is None:
        return None
    status = assignment["status"]
    done_all = status == "completed"
    started = assignment["first_activation_at"] is not None
    finished_batches = assignment["batches_done"] >= assignment["batches_total"]

    states = {
        "selected": "done",
        "assigned": "done",
        "activations": "done" if started else "active",
        "noise": ("done" if started else "pending") if noise_configured else "skipped",
        "backbone": "done" if finished_batches or done_all else ("active" if started else "pending"),
        "checkpoint": "done" if done_all else ("active" if finished_batches else "pending"),
        "completed": "done" if done_all else "pending",
    }
    if status == "failed":
        first_open = next((key for key, _ in STEP_LABELS if states[key] in ("active", "pending")), None)
        for key, _ in STEP_LABELS:
            if states[key] in ("active", "pending"):
                states[key] = "failed" if key == first_open else "pending"

    details = {
        "noise": None if noise_configured else "No noise configured yet",
        "backbone": f"{assignment['batches_done']} / {assignment['batches_total']} batches",
    }
    return {
        **_turn(assignment),
        "batches_done": assignment["batches_done"],
        "batches_total": assignment["batches_total"],
        "steps": [{"key": key, "label": label, "state": states[key], "detail": details.get(key)} for key, label in STEP_LABELS],
    }


def _turn(assignment: dict) -> dict[str, Any]:
    seconds = None
    if assignment["completed_at"]:
        seconds = (
            datetime.fromisoformat(assignment["completed_at"]) - datetime.fromisoformat(assignment["assigned_at"])
        ).total_seconds()
    return {
        "round": assignment["round_number"],
        "owner": assignment["data_owner_id"],
        "peer": assignment["compute_peer_id"],
        "status": assignment["status"],
        "backbone_version": assignment["backbone_version"],
        "version_after": assignment["backbone_version"] + 1 if assignment["status"] == "completed" else None,
        "loss": assignment["loss"],
        "accuracy": assignment["accuracy"],
        "seconds": seconds,
        "error": assignment["error"],
        "completed_at": assignment["completed_at"],
    }


def _model(db: sqlite3.Connection, session: sqlite3.Row | None, completed: list[dict]) -> dict[str, Any]:
    latest = db.execute(
        "SELECT version, created_by, created_at FROM model_checkpoints ORDER BY version DESC, created_at DESC LIMIT 1"
    ).fetchone()
    return {
        "backbone_version": latest["version"] if latest else None,
        "latest_checkpoint_at": latest["created_at"] if latest else None,
        "latest_checkpoint_by": latest["created_by"] if latest else None,
        "completed_contributions": len(completed),
        "total_turns": session["total_turns"] if session is not None else 0,
        "session_status": session["status"] if session is not None else None,
    }
