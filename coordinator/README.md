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
after each round:
  Backbone ───── evaluate.held_out ────────────>  every institution scores its own
                 <── loss / accuracy only ─────   Head + Tail on the public val split
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
| `relay/transport.py` | HTTP/HTTPS server (coordinator) and client (institutions), standard library only |
| `flower_app/protocol.py` | Message types and the `TurnInfo` sent to both parties |
| `flower_app/roles.py` | Data-owner and compute-peer logic for one turn, independent of Flower |
| `flower_app/session.py` | Coordinator loop: registration, run each turn, commit the update |
| `flower_app/observer.py` | Reports registrations, turns, batch progress and checkpoints to the metadata database for the dashboard |
| `flower_app/server_app.py`, `client_app.py` | Flower entry points |
| `flower_app/pyproject.toml` | Flower app definition and run settings |
| `flower_app/run_local.sh` | Starts 1 SuperLink and 4 SuperNodes and runs a session on one machine, over TLS |
| `flower_app/make_certs.sh` | Creates the deployment's CA and the coordinator's TLS certificate |

## Running a session locally

```bash
source venv/bin/activate
coordinator/flower_app/run_local.sh                               # 3 rounds (the default)
coordinator/flower_app/run_local.sh --run-config "num-rounds=1"   # override settings
```

- State goes to `.medweave_local/` (git-ignored): Backbone checkpoints,
  institution checkpoints, logs and Flower's config.
- A new run carries on from the latest stored Backbone. Delete
  `.medweave_local/` to start from v0.
- The first run takes about 40s and needs internet, because Flower builds a
  runtime environment. Later runs start immediately.
- Ports used: SuperLink `8000`, SuperNodes `9094`–`9097`, relay `8765`.
- **TLS:** the first run creates a CA and the coordinator's certificate in
  `.medweave_local/certs/`. The SuperLink and the relay use the certificate;
  the SuperNodes, the `flwr` CLI and the institutions verify it with `ca.crt`.
- To run in containers instead, see [`docs/docker.md`](../docs/docker.md).

Settings live in `[tool.flwr.app.config]` in `flower_app/pyproject.toml`:
`num-institutions`, `num-rounds`, `batches-per-turn`, `turn-timeout`,
`registration-timeout`, `relay-host`, `relay-port`, `relay-url`,
`partition-seed`, `batch-size`, `augment`, `augment-jitter`, `noise-std`,
`noise-seed`, `optimizer`, `learning-rate`, `weight-decay`, `lr-schedule`,
`lr-warmup`, `evaluation-split`, `evaluation-batch-size`,
`final-evaluation-split`, `tracking`. They are explained in
[`docs/coordinator_guide.md`](../docs/coordinator_guide.md), section 6.
`run_local.sh` exits 0 only when the SuperLink reports the run as
`finished:completed`.

## Tests

```bash
python -m pytest tests/unit/coordinator -v
```

The session tests call the real ClientApp handlers through a fake Flower
`Grid` and a real HTTP relay, so they cover a full session without starting
Flower processes. `run_local.sh` is the end-to-end check with real processes.

## Known limitations

- **Data and noise come from Member 3's packages.** Each data owner trains on
  its own PathMNIST shard with Gaussian activation noise (`noise-std`); see
  `docs/features/data-and-privacy.md`.
- **No disconnect handling.** A failed turn stops the session without
  changing the Backbone. Skipping or retrying an offline institution
  (SPEC §5.6) is not built yet.
- A restarted session continues from the latest Backbone but starts the
  schedule from turn 1 again.
