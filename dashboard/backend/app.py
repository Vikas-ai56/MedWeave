"""Dashboard backend: a read-only FastAPI app over the coordinator's metadata database.

GET /api/dashboard[?session_id=...]  everything the dashboard shows (latest session by default)
GET /api/sessions                    all sessions, newest first
GET /api/health                      liveness and whether the database exists
The built frontend (dashboard/frontend/dist) is served at /.
"""

import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from dashboard.backend.snapshot import build_snapshot, list_sessions
from database.store import connect_readonly

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def default_database_path() -> Path:
    """Same location the coordinator writes to (see coordinator/flower_app/server_app.py)."""
    if path := os.environ.get("MEDWEAVE_DB"):
        return Path(path)
    state = Path(os.environ.get("MEDWEAVE_STATE_DIR", Path.home() / ".medweave"))
    return state / "coordinator" / "medweave.db"


def create_app(db_path: str | Path | None = None, frontend_dist: Path | None = FRONTEND_DIST) -> FastAPI:
    db_path = Path(db_path) if db_path is not None else default_database_path()
    app = FastAPI(title="MedWeave Dashboard", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def read(fn, *args):
        if not db_path.exists():
            return None
        db = connect_readonly(db_path)
        try:
            return fn(db, *args)
        except sqlite3.OperationalError as error:  # e.g. the coordinator is creating the tables right now
            raise HTTPException(503, f"database not ready: {error}") from error
        finally:
            db.close()

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "database": str(db_path), "database_exists": db_path.exists()}

    @app.get("/api/sessions")
    def sessions() -> list[dict]:
        return read(list_sessions) or []

    @app.get("/api/dashboard")
    def dashboard(session_id: str | None = None) -> dict:
        snapshot = read(build_snapshot, session_id)
        if snapshot is None:
            return {"session": None, "institutions": [], "current_turn": None, "turns": [], "model": None,
                    "noise": None, "events": [], "database_exists": False}
        if session_id is not None and snapshot["session"] is None:
            raise HTTPException(404, f"no session {session_id!r}")
        return {**snapshot, "database_exists": True}

    if frontend_dist is not None and frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    else:
        @app.get("/", response_class=HTMLResponse)
        def frontend_missing() -> str:
            return (
                "<p>The dashboard frontend has not been built yet. Run "
                "<code>dashboard/run_dashboard.sh</code>, or <code>npm install && npm run build</code> "
                "in <code>dashboard/frontend</code>. The API is at <a href='/api/dashboard'>/api/dashboard</a>.</p>"
            )

    return app


app = create_app()
