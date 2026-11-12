# Coordinator

Member 1's module. It runs a MedWeave training session: it registers the
institutions, gives them training turns one at a time, picks a different
institution as each turn's compute peer, relays the per-batch tensors between
the two, and keeps the one canonical Backbone checkpoint.

## How a session runs

```text
coordinator (Flower ServerApp)                   institutions (Flower SuperNodes)
------------------------------                   --------------------------------
register  ───── query.register ───────────────>  each replies with its institution-id
for each turn in the schedule:
  Backbone vN ── train.peer ──────────────────>  compute peer loads Backbone vN
  start      ─── train.owner ─────────────────>  data owner starts its batches
               relay  <── activations / gradients ──>  (per batch, via HTTP)
  Backbone vN+1 <── updated weights ──────────── compute peer
  commit vN+1
```

- **Flower carries control:** registration, turn instructions, and the
  Backbone checkpoint in both directions.
- **The relay carries per-batch tensors** (SPEC §2.6). It checks each message
  against the active turn and forwards the bytes unopened. It never stores,
  decodes or logs a tensor.
- Images and labels never leave the data owner. Head and Tail checkpoints stay
  on the institution's machine.

## Layout

| Path | What it does |
|---|---|
| `scheduling/schedule.py` | Turn order: each institution trains once per round, and the next one computes for it (1→2, 2→3, 3→4, 4→1). Replaces an offline peer. |
| `checkpoint_manager/manager.py` | Stores Backbone versions `backbone_v{N}.pt`. Accepts only updates trained from the latest version. Resumes after a restart. |
| `relay/relay.py` | Relay rules and per-recipient mailboxes for the active turn |
| `relay/transport.py` | HTTP server (coordinator) and client (institutions), standard library only |
| `flower_app/protocol.py` | Message types and the `TurnInfo` sent to both parties |
| `flower_app/roles.py` | Data-owner and compute-peer logic for one turn, independent of Flower |
| `flower_app/session.py` | Coordinator loop: registration, run each turn, commit the update |
| `flower_app/server_app.py`, `client_app.py` | Flower entry points |
| `flower_app/pyproject.toml` | Flower app definition and run settings |
| `flower_app/run_local.sh` | Starts 1 SuperLink and 4 SuperNodes and runs a session on one machine |

## Running a session locally

```bash
source venv/bin/activate
coordinator/flower_app/run_local.sh                               # 1 round
coordinator/flower_app/run_local.sh --run-config "num-rounds=2"   # override settings
```

- State goes to `.medweave_local/` (git-ignored): Backbone checkpoints,
  institution checkpoints, logs and Flower's config.
- A new run carries on from the latest stored Backbone. Delete
  `.medweave_local/` to start from v0.
- The first run takes about 40s and needs internet, because Flower builds a
  runtime environment. Later runs start immediately.
- Ports used: SuperLink `8000`, SuperNodes `9094`–`9097`, relay `8765`.

Settings live in `[tool.flwr.app.config]` in `flower_app/pyproject.toml`:
`num-institutions`, `num-rounds`, `batches-per-turn`, `turn-timeout`,
`registration-timeout`, `relay-host`, `relay-port`, `relay-url`.

## Tests

```bash
python -m pytest tests/unit/coordinator -v
```

The session tests call the real ClientApp handlers through a fake Flower
`Grid` and a real HTTP relay, so they cover a full session without starting
Flower processes. `run_local.sh` is the end-to-end check with real processes.

## Known limitations

- **Placeholder data and model.** Institutions train on random tensors with
  the default `SplitModelConfig` until Member 3's loaders exist and the
  dataset is chosen.
- **No noise yet.** `perturb=None` in `client_app.py` is where Member 3's
  activation noise plugs in.
- **No TLS.** Everything runs with `--insecure` over plain HTTP. TLS is
  needed before the multi-laptop deployment.
- **No disconnect handling.** A failed turn stops the session without
  changing the Backbone. Skipping or retrying an offline institution
  (SPEC §5.6) is not built yet.
- **No metadata database.** `CheckpointRecord` and `TurnResult` hold what the
  SPEC §3.2 tables need, ready for when the database exists.
- A restarted session continues from the latest Backbone but starts the
  schedule from turn 1 again.
