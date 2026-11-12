# MedWeave: Project Implementation Report

**Repository:** `C:\Users\kartik\MedWeave` (remote `origin` = `https://github.com/Vikas-ai56/MedWeave`)
**Checked-out branch at analysis time:** `split_learning` (HEAD `90dacc4`)
**Analysis method:** Read all tracked source on both branches, read the untracked planning documents, checked Git refs and history, ran the split-learning test suite, and ran one behavioural probe. No source files were modified and no branches were switched or fetched.

### How to read this report

Each claim is tagged:

- **[Verified]**: checked in source, Git history, or by running code.
- **[Assumption]**: inferred from names or documents, not confirmed in code.

---

## 1. Project Overview

### 1.1 Purpose

MedWeave is an academic proof-of-concept for **collaborative AI training across institutions without sharing raw datasets**. It combines:

- **U-shaped split learning**: each institution keeps its input layers (Head) and output layers (Tail) and the labels; a shared middle (Backbone) runs on an assigned compute peer.
- **Sequential, rotating training turns** across four simulated institutions.
- **Configurable activation noise** on intermediate tensors before they leave an institution.
- A **coordinator** that owns the canonical Backbone checkpoint and schedules turns.

Source: `README.md` (main), `SPEC.md` §1.1 (untracked).

### 1.2 Architecture (as specified)

```
 INSTITUTION (data owner)                      COMPUTE PEER (assigned per turn)
 ------------------------                      ---------------------------------
 inputs ─> Head ─> activations ─(noise)──────> Backbone.forward
                                                    │
 labels ─> Tail <── backbone output <───────────────┘
            │
          loss ── backward
            ├──> dL/d(backbone output) ───────> Backbone.backward + optimizer step
            │                                          │
 Head.backward + optimizer step <── dL/d(activations) ┘

 COORDINATOR (Flower + custom logic): schedules turns, assigns peers,
 stores and distributes canonical Backbone checkpoint, relays payloads.
```

Source: `SPEC.md` §2.1–2.4; `split_learning/README.md` (branch).

### 1.3 Tech stack

| Layer | Specified | Actually present in repo |
| --- | --- | --- |
| ML | Python, PyTorch | **Yes** (`torch>=2.2`, `numpy`, `pytest` in `split_learning/requirements.txt`) |
| Orchestration | Flower | **No**. Not in requirements; no Flower code |
| API backend | FastAPI | **No** |
| Frontend | React | **No** |
| Database | SQLite | **No** (schema only in `SPEC.md` §3.2) |
| Containers | Docker, Docker Compose | **No** (no Dockerfile or compose file) |
| CI | GitHub Actions | **No** (no `.github/` folder) |

### 1.4 Major modules

| Module | Status |
| --- | --- |
| Split-learning engine (`split_learning/`) | **Implemented and tested** (in-process only) |
| Coordinator | **Not started** (no code) |
| Institutional client | **Not started** (only the `Institution` class in the engine) |
| Privacy / activation noise | **Hook only**. A `Callable[[Tensor], Tensor]` parameter; no noise implementation |
| Dataset loading and partitioning | **Not started** |
| Dashboard (backend and frontend) | **Not started** |
| Docker / deployment | **Not started** |

### 1.5 Overall implementation status

- **Phase 1 (Working Collaborative Training MVP, `SPEC.md` §6):** **Partially complete**. The engine covers the U-shaped math and the peer lifecycle. The coordinator, Flower orchestration, client services, rotating assignments, noise injection, and a full four-institution run are missing.
- **Phase 2 and Phase 3:** Not started.

Requirement status against `SPEC.md` §1.4:

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| FR-01 | Four independent institutional clients | Not started | No client service |
| FR-02 | Raw dataset stays local | Designed for, not enforced at system level | Engine never receives inputs at the peer (`compute_peer.py`) |
| FR-03 | U-shaped split learning | **Implemented (engine)** | `forward.py`, `backward.py`, `institution.py`, `compute_peer.py`, `training.py` |
| FR-04 | One shared Backbone | Engine-level only | `ComputePeer.load_backbone` / `backbone_update` |
| FR-05 | Different compute peer per institution | Not started | No assignment logic |
| FR-06 | Rotate assignments | Not started | No scheduler |
| FR-07 | Configurable activation noise | **Hook only** | `training.py` `perturb` parameter |
| FR-08 | Sequential coordination | Not started | `train_batches` runs one institution in one process |
| FR-09 | Store and distribute latest Backbone | Engine-level checkpoint I/O only | `checkpoint.py` |
| FR-10 | Four-institution end-to-end run | Not started | |
| FR-11 to FR-13 | Inference demo, visualisation, reconnection | Not started | |

---

## 2. Repository Layout and Folder-by-Folder Breakdown

### 2.1 Folders on disk (`main` and `split_learning` checkout)

| Path | Tracked? | Purpose | Notes |
| --- | --- | --- | --- |
| `./` (root) | Yes | Repository root | Contains `README.md`, `.gitignore`, and the two working folders |
| `split_learning/` | Yes (branch `split_learning` only) | The U-shaped split-learning engine | **Core of the project so far** |
| `tests/` | Yes (branch `split_learning` only) | Test suite | Only `tests/unit/split_learning/` exists |
| `tests/unit/split_learning/` | Yes | Unit tests for the engine | 9 files, 84 passing tests |
| `docs/` | **No** (empty on disk, untracked) | Intended docs folder | On branch `split_learning`, `docs/split_learning.md` existed until it was renamed to `split_learning/README.md` in `c96be5f`. Git does not track the empty folder on disk |
| `.git/` | n/a | Git metadata | |
| `.pytest_cache/` | No (ignored) | Pytest cache | Generated; ignored by `.gitignore` |
| `split_learning/__pycache__/` | No (ignored) | Bytecode cache | Generated |

**Folders named in the spec but absent:** `coordinator/`, `institutional_client/`, `privacy/`, `dashboard/`, `database/`, `docker/`, `plans/`. See §6.

### 2.2 Folder contents on branch `split_learning` (tracked)

```
.gitignore
README.md                        (root, identical on main)
split_learning/
  README.md                      module documentation (251 lines)
  __init__.py                    empty package marker
  backbone.py  backward.py  checkpoint.py  compute_peer.py  config.py
  errors.py    forward.py   head.py      institution.py    layers.py
  requirements.txt  tail.py  training.py  validation.py
tests/unit/split_learning/
  conftest.py  test_backward.py  test_checkpoint.py  test_compute_peer.py
  test_config.py  test_forward.py  test_institution.py  test_models.py  test_training.py
```

### 2.3 Hidden and configuration files

| File | Where | Content | Verified effect |
| --- | --- | --- | --- |
| `.gitignore` (root, `main`) | `main` (and working tree) | `__pycache__/`, `*.pyc`, `.pytest_cache/` | Correct for Python |
| `.gitignore` (root, `split_learning`) | `split_learning` only | Same three lines (added in `e4692cd`) | Identical to working tree |
| `split_learning/requirements.txt` | `split_learning` only | `torch>=2.2`, `numpy`, `pytest` | Added in `c96be5f`. Matches imports |

**Not present (verified by listing `git ls-tree -r` and `ls -la`):** `pyproject.toml`, `setup.py`, `setup.cfg`, `pytest.ini`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/`, `.env`, `.editorconfig`, `.pre-commit-config.yaml`. The only pytest configuration is `conftest.py`.

---

## 3. File-by-File Breakdown

Line counts are from the committed blobs on branch `split_learning`.

### 3.1 Root level

| File | Lines | Branches | Purpose | Status |
| --- | --- | --- | --- | --- |
| `README.md` | 144 | `main`, `split_learning` (identical) | Project overview, architecture narrative, tech stack, project tree, scope | **Partial**. Its "Run the Project" section says setup instructions "will be added". Its project tree lists folders that do not exist yet |
| `.gitignore` | 3 | `split_learning` | See §2.3 | Complete |
| `SPEC.md` | 842 | **Untracked** (working tree only) | Master product and technical specification | Planning document |
| `global_rules.md` | 114 | **Untracked** | Development rules for members and AI agents | Planning document |
| `shared_interfaces.md` | 206 | **Untracked** | Component contracts and payload shapes | Planning document |

Note: `README.md` and `global_rules.md`/`shared_interfaces.md` refer to `GLOBAL_RULES.md` and `INTERFACES.md`. The files on disk use lowercase names and are not committed. [Verified]

### 3.2 Split-learning engine (`split_learning/`)

| File | Lines | Functionality | Dependencies | Status |
| --- | --- | --- | --- | --- |
| `__init__.py` | 0 | Package marker | none | Complete |
| `config.py` | 74 | `SplitModelConfig` (frozen dataclass): input channels, image size, head/backbone conv widths, tail hidden widths, classes, seed. Derives `input_shape`, `head_output_shape`, `backbone_output_shape`. Validates widths and that the image is large enough for all pooling blocks. Also `default_optimizer` (SGD, lr 0.01), `seeded()` context manager (isolates the global RNG) | torch | Complete [Verified by tests] |
| `errors.py` | 14 | `SplitLearningError` (base), `ShapeError` (also `ValueError`), `CheckpointError`, `ProtocolError` | none | Complete |
| `layers.py` | 16 | `conv_blocks()`: repeated (3×3 conv, ReLU, 2×2 max-pool) | torch | Complete |
| `head.py` | 22 | `Head` module and `build_head(config)`. Institution-local input side | `config`, `layers` | Complete. Placeholder architecture (small CNN) |
| `backbone.py` | 22 | `Backbone` module and `build_backbone(config)`. Shared middle. Seed `config.seed + 1` | `config`, `layers` | Complete. Placeholder architecture |
| `tail.py` | 26 | `Tail` module and `build_tail(config)`. `AdaptiveAvgPool2d`, flatten, optional hidden `Linear`+ReLU, classifier. Seed `config.seed + 2` | `config` | Complete. Placeholder architecture |
| `forward.py` | 59 | Stateless forward functions `head_forward`, `backbone_forward`, `tail_forward`. Each returns a **detached clone** (what crosses the network) plus a `*Pass` object that keeps the local graph. Pass dataclasses: `HeadPass`, `BackbonePass`, `TailPass`. Tail forward takes a `loss_fn` (default cross-entropy) | torch, `head`, `backbone`, `tail` | Complete |
| `backward.py` | 41 | Stateless backward functions `tail_backward` (returns dL/d backbone output), `backbone_backward` (returns dL/d activations), `head_backward` (applies gradient to clean Head output: straight-through). Each zeroes and steps its own optimizer | torch, `forward` | Complete |
| `validation.py` | 76 | Boundary checks: `check_batch` (type, float dtype, shape, non-empty), `check_exact` (exact shape match, used for gradients), `check_labels` (long dtype, shape, range `[0, num_classes)`). `PendingBatch`: holds at most one in-flight batch per party; raises `ProtocolError` on double-put, wrong id, or take with nothing pending | torch, `config`, `errors` | Complete. See finding F-1 for a gap in Tail |
| `institution.py` | 113 | `Institution` (data owner): owns its Head, Tail, and separate Head/Tail optimizers. Methods `head_forward`, `tail_step` (returns `TailResult` with gradient and `StepMetrics`), `head_backward`, `encode`, `evaluate_tail`, `discard_pending`, `train`/`eval`. Also `StepMetrics`, `TailResult`, `metrics()`, and `create_institutions()` (identical seeds, independent parameters, rejects duplicate ids) | torch, `backward`, `forward`, `head`, `tail`, `validation` | Complete. See F-1 |
| `compute_peer.py` | 104 | `ComputePeer`: runs the Backbone for one institution's turn. `load_backbone` (resets optimizer), `forward` and `backward` (version-checked, batch-checked), `infer` (no-grad eval), `backbone_update` (returns `BackboneUpdate` with weight copy and batch count; refuses while a batch is pending), `discard_pending`, `train`/`eval`. Accepts an `optimizer_factory` | torch, `backbone`, `checkpoint`, `forward`, `backward`, `validation` | Complete |
| `checkpoint.py` | 119 | `load_module_state` (all-or-nothing with rollback), `load_optimizer_state`, `clone_state`, `read_checkpoint` (`torch.load(weights_only=True)`, error mapping). Backbone: `save_backbone_checkpoint`, `read_backbone_checkpoint` (version stored as string). Institution: `save_institution_checkpoint`, `load_institution_checkpoint` (checks institution id; rolls back Head/Tail and optimizer states on failure). Flower format: `get_weights` (NumPy list in `state_dict` order), `weights_to_state` | torch, numpy, `errors`, `institution` (type-checking only) | Complete |
| `training.py` | 108 | Single-process orchestration: `train_step` (one batch end to end; discards pending on any failure in both parties), `train_batches` (train mode, sample-weighted `Summary`), `evaluate` (eval mode, no grad, restores modes), `summarize`, `evaluation_mode` context manager. `perturb` hook applied under `no_grad` to detached activations | torch, `compute_peer`, `institution` | Complete for in-process use |
| `requirements.txt` | 3 | `torch>=2.2`, `numpy`, `pytest` | n/a | Complete for the engine. Does **not** include Flower or FastAPI |
| `README.md` | 251 | Module documentation: architecture, configuration, forward/backward contract, checkpoints, input contract, tests, limitations, and team interface notes | n/a | Complete for the engine. Its "Interface notes" list the open contract questions |

### 3.3 Tests (`tests/unit/split_learning/`)

| File | Lines | Test functions | Covers |
| --- | --- | --- | --- |
| `conftest.py` | 27 | 0 (fixtures) | Shared fixtures |
| `test_config.py` | 52 | 3 | Config validation, derived shapes |
| `test_models.py` | 27 | 3 | Head, Backbone, Tail construction and shapes |
| `test_forward.py` | 49 | 4 | Forward shapes and detachment |
| `test_backward.py` | 99 | 5 | Split gradients match a monolithic model, gradient shapes at each boundary, straight-through noise, all three components update |
| `test_institution.py` | 158 | 12 | Institution lifecycle, protocol errors, identical initial weights |
| `test_compute_peer.py` | 147 | 14 | Version checks, pending-batch rules, backbone update |
| `test_checkpoint.py` | 185 | 13 | Round-trips, atomic load, wrong-institution checks, weight transfer |
| `test_training.py` | 234 | 14 | `train_step` updates, failed-step cleanup, perturb hook, labels never reach the peer, sequential turns on one Backbone, `evaluate` updates nothing, graphs freed after each batch |

Total: **73 test functions, 84 collected test cases** (some parametrized).

**Test run (executed during this analysis):**

```
python -m pytest tests/unit/split_learning -q -p no:cacheprovider
84 passed in 2.47s
```

Environment: Python 3.13.7, torch 2.9.0+cpu, numpy 2.2.6, pytest 8.3.3, Windows. [Verified]

---

## 4. Branch-Wise Overview

### 4.1 Branch inventory

| Ref | Commit | Type | Status |
| --- | --- | --- | --- |
| `main` (local) | `396ad63` `docs: add README.md` | Local branch | Tracks `origin/main`. Contains only `README.md` |
| `split_learning` (local, **checked out**) | `90dacc4` | Local branch | 19 commits ahead of `main`, 0 behind. Contains all code |
| `origin/main` (remote-tracking) | `396ad63` | Stale local ref | **Out of date**. See §4.2 |
| `origin/split_learning` (remote-tracking) | `90dacc4` | Remote-tracking ref | Matches local `split_learning` |
| `refs/heads/main` on GitHub | `cbbe3ea` (via `git ls-remote`) | Remote only | **Not fetched.** Commit content unknown |
| `refs/heads/split_learning` on GitHub | `90dacc4` (via `git ls-remote`) | Remote only | Matches local |

No tags, stashes, or additional worktrees exist. The only worktree is the main checkout at `C:/Users/kartik/MedWeave` on `split_learning`. [Verified]

### 4.2 Remote branch status

`git ls-remote --heads origin` returned two branches: `main` and `split_learning`. [Verified]

**Finding R-1:** Remote `main` is at `cbbe3ea...`, but the local `origin/main` ref points to `396ad63`. This means `main` has been updated on GitHub after the last fetch. `cbbe3ea` is **not** in local history, so its content cannot be described here. Running `git fetch origin` would resolve this, but it was not run because the analysis must not change local refs.

### 4.3 `main` branch

- **Single commit** `396ad63` (`docs: add README.md`). The tree contains only `README.md`. [Verified]
- Describes the project and its architecture. Says setup instructions "will be added as the initial implementation is integrated" (the implementation is on `split_learning`).

### 4.4 `split_learning` branch

- 19 commits, all by one author (git user `kartik-sc`). [Verified]
- Adds the whole `split_learning/` package, its tests, `requirements.txt`, and `.gitignore` entries.
- Based directly on `main`. No merge commits. No conflicts with `main`.

**Commit classification** (by Conventional Commit prefix):

| Type | Count | Commits |
| --- | --- | --- |
| `feat` | 11 | `f61ef86`, `5606f9b`, `4a57f14`, `f6f8d87`, `1378b81`, `9f8dd2d`, `fe3d029`, `dbff734`, `d42ac69`, `5d6b29d`, `b82f589` |
| `docs` | 2 | `1b1ad8e`, `90dacc4` |
| `test` | 2 | `6ff12f1`, `715247b` |
| `fix` | 1 | `2821ecd` |
| `refactor` | 1 | `23043f4` |
| `build` | 1 | `c96be5f` |
| `chore` | 1 | `e4692cd` |

### 4.5 Unique implementations per branch

Because `split_learning` is a strict extension of `main`, the only unique content on it is the engine, tests, requirements, and `.gitignore` entries. `main` has nothing unique.

### 4.6 Git history highlights (on `split_learning`)

| Commit | Change | Significance |
| --- | --- | --- |
| `f61ef86` | Adds Head, Backbone, Tail | First model code |
| `5606f9b` | Split forward pass | Boundary detachment starts here |
| `4a57f14` | Split backward with gradient exchange | Gradient exchange starts here |
| `1378b81` | End-to-end U-shaped step | Original `step.py` |
| `9f8dd2d` | Configurable architecture and split point | Makes the split point a config choice (10 files) |
| `fe3d029` | Boundary validation, checked checkpoint loading | Adds `validation.py`, `errors.py` |
| `dbff734` | `ComputePeer` with versioned lifecycle | Peer-side state machine |
| `d42ac69` | `Institution` with independent Head and Tail | Per-party optimizers |
| `b82f589` | Training, evaluation and metrics utilities | **Deletes `step.py`**, replaced by `training.py` (4 files, +324 / −161) |
| `6ff12f1`, `715247b` | Gradient-at-boundary test; graph-freeing test | Test hardening |
| `2821ecd` | Send copies, so receivers cannot mutate senders | Correctness fix |
| `c96be5f` | Moves `docs/split_learning.md` to `split_learning/README.md`; adds `requirements.txt` | Docs and dependencies |
| `90dacc4` | Rewrites module README for integration | Latest commit |

---

## 5. Implemented Features

Everything below is verified by reading the source and, where stated, by the passing test suite.

### 5.1 Split-learning engine

**Model and configuration**
- Configurable architecture: channel widths for Head and Backbone define the split point; `tail_hidden` for the Tail; `num_classes`; `in_channels`; `image_size`; `seed`. Source: `config.py`.
- Derived boundary shapes: `head_output_shape` and `backbone_output_shape`. Source: `config.py`.
- Placeholder CNN modules. Source: `head.py`, `backbone.py`, `tail.py`, `layers.py`.
- Deterministic initialisation: each module is built with its own seed, so identical configs give identical starting weights across institutions. Source: `config.seeded`, `build_*`.

**U-shaped forward and backward**
- Institution Head forward returns a detached clone and keeps the graph locally. Source: `institution.py:50`, `forward.py`.
- Peer Backbone forward takes a fresh leaf, returns a detached clone, and keeps the graph in `PendingBatch`. Source: `compute_peer.py:53`, `forward.py`.
- Institution Tail step: loss from local labels, Tail optimizer step, returns the gradient at the Backbone output. Source: `institution.py:57`, `backward.py`.
- Peer Backbone backward: returns the gradient at the activations, steps the Backbone optimizer. Source: `compute_peer.py:61`.
- Institution Head backward: applies the returned gradient with straight-through noise handling. Source: `institution.py:68`, `backward.py`.
- Each component has its own optimizer, so stepping one never moves another. Source: `institution.py` `__init__`.

**Protocol enforcement**
- At most one batch in flight per party. Double forward raises `ProtocolError`. Source: `validation.py` `PendingBatch`.
- Wrong `batch_id` or wrong `backbone_version` raises `ProtocolError`. Source: `compute_peer.py` `_check_version`, `PendingBatch.peek`.
- Input shapes, dtypes, label dtype and range checked at every boundary. Source: `validation.py`.

**Peer lifecycle**
- `load_backbone(state_dict, version)`: loads weights, resets optimizer, clears pending state. Source: `compute_peer.py:45`.
- `backbone_update()`: returns `BackboneUpdate(base_version, state_dict copy, num_batches)`; refuses while a batch is pending. Source: `compute_peer.py:78`.
- `discard_pending()` for disconnects. `infer()` for evaluation without graph. Source: `compute_peer.py`.

**Institution lifecycle**
- `create_institutions(ids, config)`: identical initial weights, independent parameters, duplicate ids rejected. Source: `institution.py:105`.
- `encode()` and `evaluate_tail()` for evaluation. `discard_pending()`. Train and eval mode switches. Source: `institution.py`.

**Checkpoints**
- Backbone checkpoint: `{"version", "state_dict"}`, saved with `torch.save`, read with `weights_only=True`. Source: `checkpoint.py`.
- Institution checkpoint: Head, Tail, both optimizer states, and the institution id. Load checks the id and rolls back on any mismatch. Source: `checkpoint.py:71`, `:84`.
- Flower-compatible weight transfer: `get_weights` and `weights_to_state` (NumPy arrays in `state_dict` order). Source: `checkpoint.py:114`.

**Training and evaluation (single process)**
- `train_step`: one batch end to end. Discards pending state in both parties on any failure. Source: `training.py:29`.
- `train_batches`: sample-weighted loss and accuracy summary. Source: `training.py:59`.
- `evaluate`: no-grad evaluation, restores module modes afterwards. Source: `training.py:74`.
- Noise hook `perturb: Callable[[Tensor], Tensor]`, applied to detached activations. Source: `training.py:29`.

**Error model**
- `SplitLearningError` base; `ShapeError` (also `ValueError`); `CheckpointError`; `ProtocolError`. Source: `errors.py`.

### 5.2 Tests

- 84 passing tests covering shapes and detachment, gradient equality with a monolithic model, straight-through noise, checkpoint round-trips, atomic loads, protocol errors, and graph freeing between batches. [Verified by running the suite]

### 5.3 Documentation

- `split_learning/README.md` (251 lines): architecture, configuration, forward and backward contract, gradient exchange, checkpoint usage, input contract, test command, interface questions for the team, and limitations. Its claims were spot-checked against the code and the functions it names exist. [Verified in part]

### 5.4 Not implemented (no code anywhere)

- Coordinator, relay, scheduler, rotating peer assignment, canonical Backbone storage.
- Flower client or server integration.
- HTTP or wire serialization for payloads (`shared_interfaces.md` §6).
- Institutional client service, local dataset loaders, dataset partitioning.
- Activation noise implementation (only the hook exists).
- Dashboard backend and frontend, FastAPI, SQLite, database schema.
- Docker and Docker Compose, CI, TLS.
- Held-out test set and inference demo.

---

## 6. Pending Work

### 6.1 Verified defects and gaps in the engine

**F-1 (medium): `Institution.tail_step` does not enforce one Tail update per batch.**
- **Where:** `split_learning/institution.py:57` (`tail_step`) uses `self._pending.peek(batch_id)`, which does not take the batch.
- **Verified behaviour:** Calling `tail_step` twice with the same `batch_id` before `head_backward` updates the Tail weights both times. A probe script confirmed: `tail changed on first call: True`, `tail changed again on second call for same batch: True`.
- **Impact:** A retried or duplicated message from the relay would silently apply a second optimizer step, which breaks the "one batch, one update" contract the README describes.
- **Suggested fix (not applied):** Track a "tail done" state in `PendingBatch` or in `Institution`, and reject a second `tail_step` for the same batch with `ProtocolError`. Add a test next to `test_institution.py`.

**F-2 (low): `ComputePeer.backward` clears pending state before the backward pass runs.**
- **Where:** `compute_peer.py:61`. `self._pending.take(batch_id)` runs before `backbone_backward`.
- **Impact:** If the backward raises, the batch is lost. `train_step` handles this for the in-process path, but a distributed caller using `ComputePeer` directly would have no recovery path other than `discard_pending`. Minor.

**F-3 (documentation): Contract naming drift between documents and code.**
- `shared_interfaces.md` §6 defines payloads as dicts with `institution_id`, `batch_id`, `activations`, `backbone_version`, and `gradient`. The engine takes raw tensors and strings. The README acknowledges this.
- `shared_interfaces.md` says `backbone_version: str`. `SPEC.md` §3.2 says `model_checkpoints.version INTEGER`. The engine treats the version as an opaque string. The README proposes the coordinator keeps an integer and sends `str(version)`. **Unresolved.**
- `institution_id` is accepted by no method in the peer. Any cross-institution check must happen in the service layer. **Unresolved.**

**F-4 (documentation): Missing files named in documents.**
- `README.md` and `SPEC.md` refer to `GLOBAL_RULES.md`, `INTERFACES.md`, and `PLAN.md` per member. None exist. The files on disk are `global_rules.md` and `shared_interfaces.md`, both untracked and not committed.

### 6.2 Missing components relative to `SPEC.md`

| Area | Missing item | SPEC reference |
| --- | --- | --- |
| Coordinator | Whole service (scheduling, assignments, checkpoint store, relay, registration) | §2.1, §2.3, §2.4, §3.4, §5.2 |
| Coordinator | Flower integration (`coordinator/flower_app/`) | §2.5, §6 Phase 1 |
| Institutional client | `institutional_client/client.py`, `local_data.py`, `training.py` (per §3.3 layout) | §3.3 |
| Communication | HTTP APIs and serialization for activation, output, and gradient payloads | §2.6, `shared_interfaces.md` §6 |
| Privacy | `privacy/activation_noise.py`: configured noise mechanisms and scales | §2.7, FR-07 |
| Privacy | Activation perturbation metric (relative change) | §2.7 |
| Dataset | Dataset choice, loaders, four IID partitions, held-out test set | §2.8 |
| Database | `database/models.py`, migrations, SQLite schema | §3.2 |
| Dashboard | Backend and frontend | §4, FR-12 |
| Infrastructure | Dockerfiles, `docker/docker-compose.yml` | §5.2 |
| CI | GitHub Actions workflow with format, lint, test | §3.5 |
| Security | TLS configuration for multi-laptop run | §5.4 |
| Docs | `docs/architecture.md`, `docs/setup.md`, root setup instructions | §3.3, `global_rules.md` §8 |
| Process | `plans/member_N/PLAN.md` files | §3.3, §3.5 |
| Fault handling | Disconnect and reconnect handling | §5.6, FR-13 |
| Evaluation | Held-out accuracy, multi-round runs | §6 Phase 3 |

### 6.3 Placeholders and TODO markers

- Only one placeholder marker exists in tracked files: `split_learning/README.md` line 247, "The default layers are a small placeholder CNN until the dataset and task are chosen." This is intentional, documented, and tied to `SPEC.md` §2.8.
- No `TODO`, `FIXME`, `XXX`, or `NotImplemented` markers in Python source. [Verified by grep]

### 6.4 Known limitations (documented by the author)

- CPU only; no device placement.
- One batch in flight per party; sequential training only.
- Backbone optimizer state is reset every turn; only weights are checkpointed.
- Gradients sent to the peer depend on labels through the Tail. This is a known split-learning leakage channel (`SPEC.md` §1.7). Noise is a perturbation, not a privacy guarantee.

### 6.5 Recommended next steps (for the team)

1. Fix F-1 and add a regression test.
2. Freeze the payload and version types in `shared_interfaces.md` (resolve F-3).
3. Fetch `origin/main` and review commit `cbbe3ea`, which is not in local history.
4. Build the coordinator and institutional client against the engine's public functions. The engine already exposes what they need.
5. Add the noise module behind the existing `perturb` hook.
6. Add a Compose file and a CI workflow only after the in-process run passes.

---

## 7. Git Analysis

### 7.1 Commit statistics

| Ref | Commits reachable from ref |
| --- | --- |
| `main` | 1 |
| `split_learning` | 20 (1 shared with `main` + 19 new) |
| All refs combined (`git rev-list --all --count`) | 20 |

### 7.2 Branch divergence

- Merge base: `396ad63` (the tip of `main`).
- `main` ahead/behind `split_learning`: **0 / 19** (`main` has no commits that `split_learning` lacks). Equivalent to a fast-forward.
- Remote `main` (`cbbe3ea`) vs local history: **unknown**, because the commit is not in the local object store.

### 7.3 Changed-file totals on `split_learning` (versus `main`)

- Python files added on `split_learning` relative to `main`: 23 (14 in `split_learning/`, 9 in `tests/unit/split_learning/`). Plus `split_learning/README.md`, `split_learning/requirements.txt`, and a `.gitignore` change. The root `README.md` is unchanged.
- Churn across history: the largest single rewrite is `b82f589` (training utilities, +324 / −161 across 4 files). The module README was rewritten in `90dacc4` (+224 / −110).

### 7.4 Notable file history

| File | History |
| --- | --- |
| `split_learning/step.py` | Added in `1378b81`, deleted in `b82f589`. Replaced by `training.py` |
| `split_learning/README.md` | Renamed from `docs/split_learning.md` in `c96be5f`, then rewritten in `90dacc4` |
| `tests/unit/split_learning/test_train_step.py` | Added, then deleted in `b82f589`. Replaced by `test_training.py` |
| `tests/unit/split_learning/test_models.py` | Added in the initial test commit |

### 7.5 Merges and conflicts

- No merge commits. Branch history is linear.
- No stashes, tags, or extra worktrees.

---

## 8. Architecture and Dependencies

### 8.1 Module dependency graph (engine, verified from imports)

| Module | Imports from `split_learning` |
| --- | --- |
| `config`, `errors` | none |
| `layers` | none |
| `validation` | `config`, `errors` |
| `head`, `tail` | `config` (plus `layers` for `head`) |
| `backbone` | `config`, `layers` |
| `forward` | `backbone`, `head`, `tail` |
| `backward` | `forward` |
| `checkpoint` | `errors` (and `institution` under `TYPE_CHECKING` only) |
| `institution` | `backward`, `config`, `forward`, `head`, `tail`, `validation` |
| `compute_peer` | `backbone`, `backward`, `checkpoint`, `config`, `errors`, `forward`, `validation` |
| `training` | `compute_peer`, `institution` |

The runtime import graph has no cycles. `checkpoint.py` references `Institution` only in a type hint under `TYPE_CHECKING`, which avoids a cycle with `institution.py`. [Verified from import statements]

### 8.2 Data flow (one batch)

| Step | Party | Operation | Crosses boundary? |
| --- | --- | --- | --- |
| 1 | Institution | `head_forward(batch_id, inputs)`: Head forward, keep graph | No |
| 2 | Institution | Return detached clone of activations | Yes (activations →) |
| 3 | Institution (optional) | `perturb(activations)` under `no_grad` | No |
| 4 | Peer | `forward(batch_id, activations, version)`: check version and shape, Backbone forward, keep graph | No |
| 5 | Peer | Return detached clone of Backbone output | Yes (← output) |
| 6 | Institution | `tail_step(batch_id, output, labels)`: loss, Tail step, return gradient | No (labels stay) |
| 7 | Institution | Return gradient at Backbone output | Yes (gradient →) |
| 8 | Peer | `backward(batch_id, gradient, version)`: Backbone step, return gradient at activations | No |
| 9 | Peer | Return gradient at activations | Yes (← gradient) |
| 10 | Institution | `head_backward(batch_id, gradient)`: Head step, release batch | No |

Only steps 2, 5, 7, and 9 carry tensors across the boundary. Labels (step 6) and raw inputs (step 1) never leave the institution. [Verified in code]

### 8.3 Checkpoint flow

```
Coordinator (canonical Backbone, version V)
   │  state_dict + version (save_backbone_checkpoint / read_backbone_checkpoint)
   ▼
Compute peer: load_backbone(state, V) ──> turn ──> backbone_update() ──> (version V, state, n)
   │
   ▼
Coordinator stores as V+1 (not implemented)

Institution: Head + Tail + optimizers, saved locally with save_institution_checkpoint.
```

### 8.4 External dependencies

| Package | Used by | Declared in |
| --- | --- | --- |
| `torch` (>=2.2; tested with 2.9.0+cpu) | All engine modules | `requirements.txt` |
| `numpy` | `checkpoint.py` (weight transfer) | `requirements.txt` |
| `pytest` | Tests | `requirements.txt` |

Not used, not declared: Flower, FastAPI, SQLite drivers, Docker, React. [Verified]

### 8.5 Interaction with planned services (not yet built)

- The **coordinator** would call `load_backbone`, `backbone_update`, and the checkpoint readers and writers.
- The **institutional client** would call `create_institutions` or `Institution`, `head_forward`, `tail_step`, `head_backward`, `save_institution_checkpoint`.
- The **peer service** would wrap `ComputePeer.forward`, `backward`, `infer`, and `backbone_update` behind HTTP.
- The **privacy layer** would provide a `Callable[[Tensor], Tensor]` for the `perturb` argument.

The engine's docstrings and `split_learning/README.md` state these boundaries explicitly.

---

## 9. Assumptions and Limits of This Analysis

- **Verified:** all tracked files on `main` and `split_learning`; untracked planning files; Git refs and history; `git ls-remote` output for remote heads; test results; the probe in F-1.
- **Not verified:** the content of remote `main` at `cbbe3ea` (not fetched, by instruction).
- **Not verified:** whether the GitHub repository has other branches not reported by `git ls-remote --heads` (only heads were queried; no tags or PR refs).
- **Assumption:** the untracked planning files represent the team's current intent. They are not committed, so the team may not have agreed to them.
- **Not run:** Docker, the frontend, or any end-to-end test, because none exist.
