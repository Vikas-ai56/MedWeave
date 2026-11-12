# Dashboard

A live view of a MedWeave training session (SPEC §4). The page is read-only
and refreshes every 1.5 seconds.

The interface is "The Living Weave": a dark observatory where institutions sit
inside their own data boundaries around a turning, geodesic Shared Backbone.
Five views, all hash routes (`#/weave`, `#/experiments`, ...), so the backend
needs no new routes:

| View | Shows |
|---|---|
| **The Weave** | Institutions around the Shared Backbone. Noised activations pulse from the data owner to the core and gradients return; the compute peer's thread shows it hosting the Backbone. The 7 turn steps sit in an execution rail. The readout gives round, turns, connected institutions, data owner, compute peer, Backbone version, latest training accuracy, noise and "raw data moved: none". Hover an institution to focus it; click it or the core to open a floating detail layer with a progressive "View evidence" section. A completed turn plays a short (<1 s) reweave transition. |
| **Experiments** | The research record: validation accuracy by round for the centralized reference and the three split-learning steps (from the curated MLflow runs), test metrics, the noise sweep (privacy cost) and bytes per round computed from the model's tensor shapes. Below: this session's training loss and accuracy per finished turn, and the full turn history (status filter, CSV export of the view). |
| **Institutions** | The same institutions as a constellation, in contribution, performance or protection mode, and a ledger with one card per institution: status, role, batch progress, latest contribution, counts, last seen. Filters (All / Active / Owner / Peer / Idle) change the view only. |
| **Protection** | Local data → protected update → shared model, step by step; what the coordinator can and cannot see; the three privacy placeholders with the noise configuration; and the limits, stated plainly. |
| **Evidence** | Proof rail (the current turn's steps), the Backbone as a model artifact, the activity log (latest 40 events, type filter, jump to newest), participation per institution, the measured centralized-vs-split comparison, attack tests, and SDG 9 indicators. |

Every status is an icon plus a label, never colour alone. Motion respects
`prefers-reduced-motion`: pulses become static direction arrows and the core stops turning.

### Simulated session

With no coordinator running, **Simulated** in the command bar (or "Preview a simulated
session" on the empty and offline screens) plays a generated session in the browser:
four institutions, 10 rounds, one failed turn. `src/lib/simulate.js` is a pure function of
time that returns the exact `/api/dashboard` shape and mirrors `snapshot.py`'s turn steps
and roles, so every view renders it through the same code path as live data. It is
labelled "Simulated" in amber wherever the connection status appears and is never mixed
with live data.

### What the interface deliberately does not claim

The design brief asked for an ε privacy budget, clipping, secure aggregation, on-chain
audit records and a FedAvg comparison. MedWeave has none of these (SPEC §1.7,
`privacy/noise.py`), so the interface shows what exists instead: the noise σ, the
coordinator's own event log, a "coordinator-visible / stays at the institution" boundary,
and FedAvg marked "not measured". Its communication cost is shown only as an analytical
estimate.

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
| `frontend/src/App.jsx` | Shell, hash routing, polling, live/simulated source, session selector, refresh |
| `frontend/src/usePolling.js` | Polls a URL, keeping the last good response when a request fails |
| `frontend/src/views/` | One file per view: Weave, Experiments, Institutions, Protection, Evidence |
| `frontend/src/components/weave/` | The canvas (`WeaveCanvas.jsx`) and the geodesic Backbone (`Core.jsx`, rotated in an animation frame outside React) |
| `frontend/src/components/Panels.jsx` | Overview readout, institution cards, execution rail, model artifact, turn table, privacy tiles, event stream |
| `frontend/src/components/DetailLayer.jsx` | The floating institution / global model layer |
| `frontend/src/components/LineChart.jsx` | Per-turn loss and accuracy (SVG, hover and arrow-key crosshair, failed turns as hollow markers) |
| `frontend/src/components/SeriesChart.jsx` | Multi-series research chart with direct labels and marker shapes |
| `frontend/src/components/Status.jsx` | Status indicators: always an icon plus a label |
| `frontend/src/lib/` | Pure logic: derived values, simulator, research record, geometry, CSV export, hooks |
| `frontend/src/styles/` | Design tokens and styles, one file per layer (imported by `styles.css`) |
| `../database/schema.sql`, `../database/store.py` | The SQLite tables and the coordinator's writer |

## Tests

```bash
python -m pytest tests/unit/dashboard tests/unit/database tests/unit/coordinator/test_observer.py -v
```

## Dependencies

- **Backend:** FastAPI and uvicorn, which are already installed with Flower.
- **Frontend:** React 19 and Vite 6, pinned in `frontend/package.json`. Fonts
  (Newsreader, Geist, Geist Mono) are bundled through `@fontsource-variable`, so the
  dashboard renders correctly with no internet connection. No chart or animation
  libraries: charts are hand-written SVG, and motion is CSS and SVG `animateMotion`.
- **Research record:** `frontend/src/lib/research.js` holds numbers copied from the
  curated MLflow runs and the noise sweep, with their sources. Update it when those runs change.

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
