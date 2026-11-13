# Dashboard

A live view of a MedWeave training session (SPEC §4). The page is read-only
and refreshes every 1.5 seconds.

| Panel | Shows |
|---|---|
| **Overview** | Round, completed turns, connected institutions, current data owner and compute peer, Backbone version |
| **Institution network** | One card per institution: online/offline/busy, role (data owner / compute peer / idle), live batch progress, latest contribution |
| **Training activity** | The 7 steps of the current turn (SPEC §4.3 C), plus the shared Backbone's version and latest checkpoint |
| **Training metrics** | Loss and accuracy for each finished turn, and a turn history table with the same data |
| **Privacy** | Noise configuration. The before/after activation view arrives with the privacy layer (later phase) |
| **Activity log** | The latest 40 events |

## How it gets its data

```text
coordinator (server_app.py)
  └─ DatabaseObserver ──writes──> .medweave_local/coordinator/medweave.db  (SQLite, SPEC §3.2)
                                         │ read-only
dashboard/backend (FastAPI) ─────────────┘──> /api/dashboard ──> dashboard/frontend (React)
```

- The coordinator records registrations, turns, batch progress, checkpoints
  and events as they happen. Batch progress comes from the relay counting
  messages; it never opens them.
- The backend only reads the database. The database never holds images,
  labels, activations or gradients, so the dashboard can't show them.

## Running it

From the repo root, in a second terminal next to `run_local.sh`:

```bash
dashboard/run_dashboard.sh              # http://127.0.0.1:8080
dashboard/run_dashboard.sh --rebuild    # rebuild the frontend after changing it
PORT=9000 dashboard/run_dashboard.sh    # another port
```

- The first start runs `npm install` and `npm run build` in
  `dashboard/frontend`, which needs Node.js.
- The dashboard can be started before, during or after a session. It shows
  the latest session.
- Both scripts use `MEDWEAVE_STATE_DIR` (default `.medweave_local/`) to find
  the database.

### Working on the frontend

```bash
dashboard/run_dashboard.sh                      # terminal 1: the API on :8080
cd dashboard/frontend && npm run dev            # terminal 2: hot reload on :5173 (forwards /api to :8080)
```

## API

| Endpoint | Returns |
|---|---|
| `GET /api/dashboard` | Everything the page shows, for the latest session |
| `GET /api/dashboard?session_id=...` | Same, for a specific session |
| `GET /api/sessions` | All sessions, newest first |
| `GET /api/health` | Liveness and the database path |
| `GET /api/docs` | Interactive API docs |

## Files

| Path | What it does |
|---|---|
| `backend/app.py` | FastAPI app. Serves the API and the built frontend |
| `backend/snapshot.py` | Turns database rows into the dashboard's view (roles, turn steps, history) |
| `frontend/src/App.jsx` | Page layout and polling |
| `frontend/src/components/Panels.jsx` | Overview, institution cards, turn steps, model status, table, privacy, log |
| `frontend/src/components/LineChart.jsx` | Loss and accuracy charts (SVG, hover crosshair) |
| `frontend/src/components/Status.jsx` | Status indicators: always an icon plus a label |
| `../database/schema.sql`, `../database/store.py` | The SQLite tables and the coordinator's writer |

## Tests

```bash
python -m pytest tests/unit/dashboard tests/unit/database tests/unit/coordinator/test_observer.py -v
```

## Dependencies

- **Backend:** FastAPI and uvicorn, which are already installed with Flower.
- **Frontend:** React 19 and Vite 6, pinned in `frontend/package.json`.

## Known limitations

- **Polling, not push.** Updates can lag by up to 1.5 seconds.
- **Offline is detected only between turns.** The coordinator checks Flower's
  connected nodes before each turn.
- **The privacy panel is a placeholder** until the noise layer and its
  activation samples exist. The coordinator must not relay raw images, so the
  "original image" view must be served by the institution itself.
- **No login.** It listens on `127.0.0.1` only. Do not expose it on a network
  as it is.
- **Metrics are training metrics** on the data owner's batches, not held-out
  test accuracy.
