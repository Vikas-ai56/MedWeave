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
| Dataset loading and activation noise | Member 3 | **Not yet.** Random placeholder data, no noise |
| Dashboard and metadata database | Member 4 (built by Member 1) | Done: live dashboard, see [`dashboard/README.md`](../dashboard/README.md) |
| Docker | Member 4 | **Not yet** |

Because the data is random, the loss stays around 0.69 and accuracy around
50%. The run proves the workflow, not the learning.

## 2. Setup

The code is on the `coordinator` branch, which already includes Member 2's
`split_learning` code.

```bash
git fetch
git checkout coordinator

python3 -m venv venv                  # must be called "venv", at the repo root
source venv/bin/activate
pip install torch numpy flwr pytest
```

Tested with Python 3.14.7, torch 2.15 and flwr 1.43. The run script looks for
the virtual environment at `venv/` in the repo root.

## 3. Run a session

From the repo root:

```bash
coordinator/flower_app/run_local.sh                               # 1 round
coordinator/flower_app/run_local.sh --run-config "num-rounds=2"   # 2 rounds
```

To watch it live, run `dashboard/run_dashboard.sh` in a second terminal and
open http://127.0.0.1:8080.

Expected output (other log lines removed):

```
[coordinator] registered institutions: inst-1, inst-2, inst-3, inst-4
[coordinator] initialised Backbone v0
[coordinator] round 1: inst-1 trains, inst-2 computes, Backbone v0
[coordinator]   done in 6.1s: 5 batches, loss 0.686, accuracy 0.57 -> Backbone v1
[coordinator] round 1: inst-2 trains, inst-3 computes, Backbone v1
...
[coordinator] session complete: coordinator holds Backbone v4
```

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

  per batch, through the relay (HTTP):
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

**Data.** Replace the call to `placeholder_batches(...)` with your loader. It
must return a list (or other iterable) of `(inputs, labels)` per batch:

| Tensor | Shape | dtype |
|---|---|---|
| `inputs` | `(B, C, H, W)` | float |
| `labels` | `(B,)`, values in `0 .. num_classes-1` | `torch.long` |

Each institution should load only its own partition. The institution's id is
available as `me` (`"inst-1"`, ...).

**Noise.** Replace `perturb=None` with a function
`Callable[[Tensor], Tensor]`:

- It receives the Head's activations, shape `(B, *MODEL_CONFIG.head_output_shape)`.
- It must return a tensor of the **same shape**.
- It is applied just before the activations are sent. Gradients pass straight
  through it, which is exact for additive noise.

**Model shape.** When the dataset is chosen, update `MODEL_CONFIG` in
`coordinator/flower_app/protocol.py` to match: `in_channels`, `image_size`
and `num_classes`. It is currently `SplitModelConfig()`: grayscale, 28×28,
2 classes. Every party reads this one value, so they always agree.

**Batches per turn.** Set `batches-per-turn` (see section 6) to match your
partition size.

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

**Docker, and later multiple laptops.**

Processes to run:

| Process | Command | Port |
|---|---|---|
| Coordinator | `flower-superlink --insecure` | 8000 |
| Relay | started by the coordinator automatically | 8765 |
| Each institution | `flower-supernode --insecure --superlink <coordinator>:8000 --port 9094 --node-config 'institution-id="inst-N"'` | 9094 |

Each institution only needs a distinct port when several run on the same
machine.

Environment for every process:

| Variable | Purpose |
|---|---|
| `PYTHONPATH` | the repo root |
| `PATH` | must include the venv's `bin` |
| `MEDWEAVE_STATE_DIR` | where that machine's checkpoints go |
| `FLWR_HOME` | where Flower keeps its config |

Settings to change in `pyproject.toml` for multi-laptop runs:
- `relay-host`: the address the relay listens on, e.g. `0.0.0.0` to accept
  connections from other machines.
- `relay-url`: the address the institutions use to reach the relay. It must
  be the coordinator laptop's LAN address, e.g. `http://192.168.1.10:8765`,
  not `127.0.0.1`.

**TLS.** Everything currently runs with `--insecure` over plain HTTP. TLS
must be added before the multi-laptop demo (SPEC §5.4).

## 6. Settings

Settings are in `[tool.flwr.app.config]` in
`coordinator/flower_app/pyproject.toml`. To override one for a single run:
`run_local.sh --run-config "key=value"`.

| Setting | Default | Meaning |
|---|---|---|
| `num-institutions` | 4 | How many institutions the coordinator waits for |
| `num-rounds` | 1 | Each round gives every institution one turn |
| `batches-per-turn` | 5 | Batches each data owner trains per turn |
| `turn-timeout` | 300.0 | Seconds before a turn is treated as failed |
| `registration-timeout` | 120.0 | Seconds to wait for all institutions to connect |
| `relay-host` | `127.0.0.1` | Address the relay listens on |
| `relay-port` | 8765 | Relay port |
| `relay-url` | `http://127.0.0.1:8765` | Address the institutions use to reach the relay |

## 7. Tests

```bash
python -m pytest tests/unit -v               # everything (162 tests)
python -m pytest tests/unit/coordinator -v   # coordinator only
```

The session tests run a full 4-institution session through the real
institution code and a real HTTP relay, without starting Flower processes.
`run_local.sh` is the end-to-end check with real Flower processes.

## 8. Known limitations

- **Placeholder data and model** until Member 3's loader exists and the
  dataset is chosen.
- **No noise yet.** The `perturb` hook is ready (section 5).
- **No TLS.**
- **No disconnect handling.** If a turn fails, the session stops and the
  Backbone is left unchanged. Skipping or retrying an offline institution is
  not built yet.
- **No held-out test accuracy.** Each turn reports training loss and accuracy
  only.
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
| Want a clean start from v0 | `rm -rf .medweave_local` |
| Run is slow the first time | Normal. Flower builds its environment once (about 40 seconds). |
