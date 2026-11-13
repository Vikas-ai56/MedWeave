# Coordinator Guide for Teammates

How to run the MedWeave coordinator and how your module connects to it.
Maintained by Member 1. For the file-by-file layout, see
[`coordinator/README.md`](../coordinator/README.md).

## 1. What works today

The coordinator runs a full Phase 1 session on one laptop:

1. 4 institutions register.
2. Each institution takes a training turn, and a different institution runs
   the Backbone for it.
3. The coordinator keeps every Backbone version and ends up holding the final
   one.

| Part | Owner | Status |
|---|---|---|
| Split-learning engine (Head, Backbone, Tail) | Member 2 | Done, used by the coordinator |
| Scheduler, checkpoints, relay, Flower orchestration | Member 1 | Done, tested end to end |
| Dataset loading and activation noise | Member 3 | Done: PathMNIST shards per institution, Gaussian activation noise (`docs/features/data-and-privacy.md`) |
| Dashboard and metadata database | Member 4 (built by Member 1, redesigned by Member 2) | Done: live dashboard, see [`dashboard/README.md`](../dashboard/README.md) |
| Docker and TLS | Member 4 (built by Member 1) | Done: `docker compose`, see [`docs/docker.md`](docker.md); TLS in every run |

The demo settings (all defaults in `pyproject.toml`):
- **Model:** MedWeavePathCNN-v1 with GroupNorm (Member 2).
- **Data:** each institution trains on its own PathMNIST shard (9 classes), with flips, rotations and
  colour jitter.
- **Training:** AdamW (weight decay 1e-4), with a cosine learning rate that the coordinator sets per turn.
- **Privacy:** activation noise 0.15.
- **Security:** everything over TLS.

Each turn is one pass over the owner's shard (351 batches of 64, about 22 s). After every round each
institution scores its own Head and Tail on the validation split, and after the last round on the test
split.

**Measured on 2026-11-13** (3 rounds, about 5.5 minutes):

| Round | Mean validation accuracy |
|---|---|
| 1 | 0.704 |
| 2 | 0.777 |
| 3 | 0.798 |

The final mean test accuracy was **0.831** (institutions 0.825–0.835).

## 2. Setup

Everything is merged on the `integration/demo` branch: the coordinator, Member 2's engine and model,
Member 3's data and noise, and the dashboard.

```bash
git fetch
git checkout integration/demo

python3 -m venv venv                  # must be called "venv", at the repo root
source venv/bin/activate
pip install -r requirements.txt pytest
```

To use Docker instead of a virtual environment, see [`docs/docker.md`](docker.md).

The first `run_local.sh` also downloads PathMNIST (205 MB, once per machine,
into `~/.medweave/data` or `$MEDWEAVE_DATA_DIR`). To do that ahead of a demo:
`python -m institution_data prepare`.

Tested with Python 3.14.7, torch 2.15 and flwr 1.43. The run script looks for
the virtual environment at `venv/` in the repo root.

## 3. Run a session

From the repo root:

```bash
coordinator/flower_app/run_local.sh                               # 3 rounds (the demo)
coordinator/flower_app/run_local.sh --run-config "num-rounds=1"   # a quicker run
```

To watch it live, run `dashboard/run_dashboard.sh` in a second terminal and
open http://127.0.0.1:8080.

Expected output (other log lines removed):

```
[coordinator] registered institutions: inst-1, inst-2, inst-3, inst-4
[coordinator] initialised Backbone v0
[coordinator] round 1: inst-1 trains, inst-2 computes, Backbone v0, learning rate 1.00e-03
[coordinator]   done in 21.2s: 351 batches, loss 1.435, accuracy 0.46, noise std 0.15 (activations changed 17.6%) -> Backbone v1
...
[coordinator] round 3 val accuracy with Backbone v12 (10004 samples): inst-1 0.776, inst-2 0.792, inst-3 0.809, inst-4 0.813; mean 0.798
[coordinator] session complete: coordinator holds Backbone v12
[coordinator] round 3 test accuracy with Backbone v12 (7180 samples): inst-1 0.825, inst-2 0.828, inst-3 0.835, inst-4 0.834; mean 0.831
run 7055488590190374045: finished:completed
```

The script's exit code follows the last line: 0 only for `finished:completed`.
A failed turn or evaluation ends the run as `finished:failed` and exit 1.

Good to know:

- **The first run takes about 40 seconds and needs internet.** Flower builds a
  runtime environment the first time. Later runs start straight away, so do a
  practice run before any demo.
- **Everything is saved in `.medweave_local/`** (git-ignored):

  | Folder | Contents |
  |---|---|
  | `coordinator/checkpoints/` | `backbone_v0.pt`, `backbone_v1.pt`, ... |
  | `institutions/` | `inst-1.pt` ... each institution's own Head and Tail |
  | `logs/` | `superlink.log`, `inst-1.log` ... |
  | `certs/` | the deployment's CA and the coordinator's TLS certificate |

- **A new run continues from the latest saved Backbone.** To start over from
  v0, run `rm -rf .medweave_local`.

## 4. How a session works

The script starts 6 kinds of processes on your laptop:

- 1 **SuperLink**: Flower's server, port 8000
- 4 **SuperNodes**: one per institution, ports 9094–9097, each started with
  `institution-id="inst-N"`
- the **coordinator** itself (Flower's ServerApp), which also runs the relay
  on port 8765

```text
COORDINATOR                                            INSTITUTIONS
register ─────────── "query.register" ──────────────>  each replies with its id
for each turn (e.g. inst-1 trains, inst-2 computes):
  Backbone vN ─────── "train.peer" ────────────────>  inst-2 loads Backbone vN
  start ───────────── "train.owner" ───────────────>  inst-1 starts its batches

  per batch, through the relay (HTTPS):
    inst-1 ── activations ──────> inst-2   (Backbone forward)
    inst-1 <── backbone output ── inst-2
    inst-1 ── gradient ─────────> inst-2   (Backbone backward + update)
    inst-1 <── input gradient ─── inst-2
  inst-1 ── done ──> inst-2

  Backbone vN+1 <──── updated weights ─────────────── inst-2
  save as vN+1, then next turn
```

There are two kinds of traffic:

- **Flower handles control:** registration, "start your turn", and sending
  the Backbone out and back.
- **The relay handles the per-batch tensors.** It checks every message
  belongs to the current turn (right institutions, right Backbone version),
  forwards it without opening it, and never stores or logs it.

What stays where:

- Images and labels never leave the data owner.
- Each institution's Head and Tail stay on its own machine.
- The compute peer only ever sees noisy activations and gradients.

## 5. How your module plugs in

### Member 3: data and noise

Both connection points are in `coordinator/flower_app/client_app.py`, in
`train_own_data()`.

Done. With `augment = true` the batches also get flips, rotations and colour
jitter (`institution_data/augment.py`), seeded per institution and round. In
`train_own_data()` the data owner opens only its own shard
(`institution_data.open_institution_training`) and builds the turn's batches
from `TurnInfo.round_number` (`training_loader`): float `(B, 3, 28, 28)` in
[0, 1] and `torch.long` `(B,)` labels in 0..8. It perturbs the activations with
`privacy.GaussianActivationNoise` before sending them and reports `noise_std`
and `activation_change_pct` with its metrics. `MODEL_CONFIG` is PathMNIST:
3×28×28, 9 classes. Details: `docs/features/data-and-privacy.md`. The noise
levels were re-measured for the new model (section 1 of that document).

### Member 2: split-learning engine

The coordinator uses these parts of `split_learning`:

- `ComputePeer`: `load_backbone`, `forward`, `backward`, `backbone_update`,
  `discard_pending`
- `Institution`: `head_forward`, `tail_step`, `head_backward`,
  `discard_pending`
- `split_learning.checkpoint`: Backbone and institution checkpoint save/load
- `training.summarize` and `Summary`

If you change any of these signatures, tell Member 1. The test
`tests/unit/coordinator/test_roles.py` checks that a turn through the relay
gives exactly the same loss and weights as your in-process
`train_batches`, so it will catch most breaking changes.

### Member 4: dashboard, database and Docker

**Dashboard and database (already built).**
- The coordinator writes every session to SQLite at
  `.medweave_local/coordinator/medweave.db`, using the SPEC §3.2 tables in
  `database/schema.sql`.
- `dashboard/run_dashboard.sh` serves the live dashboard at
  http://127.0.0.1:8080.
- To show more on the dashboard, add a hook to `SessionObserver` in
  `coordinator/flower_app/observer.py`, then read the new data in
  `dashboard/backend/snapshot.py`. Details are in
  [`dashboard/README.md`](../dashboard/README.md).

**Results in code.**
- `CheckpointManager.commit()` returns a `CheckpointRecord` with `version`,
  `storage_path`, `created_by` and `created_at`.
- `run_session()` returns a `SessionResult`: `turns` holds a `TurnResult` per
  turn with `turn` (round, data owner, compute peer), `checkpoint`, `loss`,
  `accuracy`, `num_batches`, `seconds`, `noise_std` and
  `activation_change_pct`; `evaluations` holds a `RoundEvaluation` per round
  with each institution's held-out `accuracy` and `loss`.
- Every session is also recorded in MLflow (`docs/features/experiment-tracking.md`).

**Docker** is done: see [`docs/docker.md`](docker.md).

**Multiple laptops.** One laptop is the coordinator; each other laptop is one institution.

1. On the coordinator laptop, create the TLS files. Name the laptop's LAN address so institutions can
   verify it:

   ```bash
   coordinator/flower_app/make_certs.sh certs 192.168.1.10
   ```

   Copy `certs/ca.crt`, and only that file, to every institution laptop.
2. Start the processes:

   | Laptop | Command |
   |---|---|
   | Coordinator | `flower-superlink --host 0.0.0.0 --ssl-ca-certfile certs/ca.crt --ssl-certfile certs/server.crt --ssl-keyfile certs/server.key` |
   | Each institution | `flower-supernode --root-certificates ca.crt --superlink 192.168.1.10:8000 --node-config 'institution-id="inst-N"'` |

3. Set the environment variables:

   | Variable | On | Purpose |
   |---|---|---|
   | `PYTHONPATH` | every laptop | the repo root |
   | `PATH` | every laptop | must include the venv's `bin` |
   | `MEDWEAVE_STATE_DIR` | every laptop | where that machine's checkpoints go |
   | `MEDWEAVE_DATA_DIR` | institutions | where that machine's shard is (`python -m institution_data prepare`) |
   | `MEDWEAVE_TLS_CERT`, `MEDWEAVE_TLS_KEY` | coordinator | `certs/server.crt` and `certs/server.key`, for the relay |
   | `MEDWEAVE_TLS_CA` | institutions | `ca.crt`, to verify the relay |
   | `FLWR_HOME` | where you run `flwr run` | its `config.toml` needs `root-certificates` pointing at `ca.crt` |

4. Start the session with these settings:

   ```bash
   --run-config 'relay-host="0.0.0.0" relay-url="https://192.168.1.10:8765"'
   ```

   - `relay-host="0.0.0.0"` lets the relay accept other machines.
   - `relay-url` must use the coordinator's LAN address, the same one named in the certificate.

## 6. Settings

Settings are in `[tool.flwr.app.config]` in
`coordinator/flower_app/pyproject.toml`. To override one for a single run:
`run_local.sh --run-config "key=value"`.

| Setting | Default | Meaning |
|---|---|---|
| `num-institutions` | 4 | How many institutions the coordinator waits for |
| `num-rounds` | 3 | Each round gives every institution one turn |
| `batches-per-turn` | 351 | Batches each data owner trains per turn (351 × 64 = one pass over a 22,499-sample shard) |
| `turn-timeout` | 300.0 | Seconds before a turn is treated as failed |
| `registration-timeout` | 120.0 | Seconds to wait for all institutions to connect |
| `relay-host` | `127.0.0.1` | Address the relay listens on |
| `relay-port` | 8765 | Relay port |
| `relay-url` | `https://127.0.0.1:8765` | Address the institutions use to reach the relay; `https://` needs `MEDWEAVE_TLS_CERT`/`MEDWEAVE_TLS_KEY` on the coordinator |
| `partition-seed` | 0 | Which provisioned PathMNIST partition and batch order |
| `batch-size` | 64 | Training batch size |
| `augment` | `true` | Flips, 90° rotations and colour jitter on training batches |
| `augment-jitter` | 0.2 | Colour jitter strength (0 = orientation only) |
| `noise-std` | 0.15 | Gaussian noise std on Head activations (0 = no noise); chosen from the noise sweep |
| `noise-seed` | 0 | Base seed for the noise |
| `optimizer` | `adamw` | Optimizer for Head, Tail and Backbone (`adamw`, `adam` or `sgd`) |
| `learning-rate` | 0.001 | Base learning rate |
| `weight-decay` | 0.0001 | Weight decay for that optimizer |
| `lr-schedule` | `cosine` | `cosine` (warm-up, then decay to 0 over the session's turns) or `constant`; one rate per turn, sent in `TurnInfo` |
| `lr-warmup` | 0.05 | Fraction of the turns used for linear warm-up |
| `evaluation-split` | `val` | Held-out split each institution scores after every round (`val`, `test`, or `none` to skip) |
| `evaluation-batch-size` | 512 | Batch size for that evaluation |
| `final-evaluation-split` | `test` | Split every institution scores once after the last round (`none` to skip) |
| `tracking` | `mlflow` | Record the session in MLflow (`python -m tracking ui` to view), or `none` |

## 7. Tests

```bash
python -m pytest tests -v                    # everything (306 tests)
python -m pytest tests/unit/coordinator -v   # coordinator only
```

The session tests run a full 4-institution session through the real
institution code and a real HTTP relay, without starting Flower processes.
`run_local.sh` is the end-to-end check with real Flower processes.

## 8. Known limitations

- **No disconnect handling.** If a turn fails, the session stops and the
  Backbone is left unchanged. Skipping or retrying an offline institution is
  not built yet.
- **Gradients can reveal labels.** Activation noise does not protect the
  gradient sent back to the compute peer (`docs/features/data-and-privacy.md`).
- **Not differential privacy.** `noise-std` is not an epsilon.
- A restarted session continues from the latest Backbone but starts the turn
  order from the first turn again.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `No such file or directory: 'flower-superexec'` | The venv isn't on your `PATH`. Use `run_local.sh`, or run `source venv/bin/activate` first. |
| `Connection to the SuperLink is unavailable` | The SuperLink didn't start. Check `.medweave_local/logs/superlink.log`. Port 8000 may be taken by a leftover process. |
| `Address already in use` (8000, 8765, 9094–9097) | Flower processes from an earlier run are still alive. Run `pkill -f "venv/bin/flower-"`. They can take a few seconds to shut down on their own. |
| `only N of 4 institutions connected` | A SuperNode failed to start. Check `.medweave_local/logs/inst-N.log`. |
| `turn ... not committed: ...` | The message lists what went wrong on **both** sides. Usually one side's real error caused the other side to time out. |
| `StaleUpdateError` | A Backbone update was trained from an older version than the latest. Nothing was saved. |
| `no partition manifest` | The institution's data is not provisioned for that `partition-seed`. Run `python -m institution_data prepare --seed N`. |
| `checkpoint was trained with SGD, institution uses Adam` | The `optimizer` setting changed since the institution's last checkpoint. Start fresh: `rm -rf .medweave_local`. |
| `Backbone checkpoint does not match the configured architecture` | The model changed (e.g. GroupNorm was added) since the stored checkpoints were made. Run `rm -rf .medweave_local`. |
| Want a clean start from v0 | `rm -rf .medweave_local` |
| Run is slow the first time | Normal. Flower builds its environment once (about 40 seconds). |
| `CERTIFICATE_VERIFY_FAILED` or `cannot reach relay` | The certificate does not name the address in use, or the institution has the wrong `ca.crt`. Re-create with `make_certs.sh <dir> <LAN IP>` and copy the new `ca.crt`. |
| `relay-url ... does not match the relay's TLS setting` | Use `https://` with `MEDWEAVE_TLS_CERT`/`MEDWEAVE_TLS_KEY` set (as `run_local.sh` does), or `http://` without them. |
