# Plan: integrate data-privacy, coordinator and the split-learning CNN into one demo

## Context

Three branches each hold part of the MedWeave demo, and none can run it alone:

| Branch | Owner | Has | Missing |
| --- | --- | --- | --- |
| `feature/data-privacy` (`0d98fdb`) | Member 3, on top of coordinator `6511f01` | Real Flower session (`run_local.sh`: SuperLink + 4 SuperNodes + relay), PathMNIST shards per institution, Gaussian activation noise, per-round held-out evaluation, MLflow `tracking/` package, noise comparison | Our CNN (still the 2-layer placeholder, about 0.76 validation accuracy), augmentation, LR schedule, the dashboard |
| `coordinator` (`d8c7627`) | Member 1 / 4 | Everything above up to `6511f01`, plus the live dashboard (`dashboard/`) and metadata database (`database/`) | Data, noise, tracking |
| `split_learning` (`28f9477`) | Member 2 | MedWeavePathCNN-v1/v2, BatchNorm/GroupNorm options, PathMNIST experiment harness, best recipe (test accuracy 0.883 ± 0.009, AUC 0.988, split across 4 institutions) | Flower, real shards, noise, dashboard |

`feature/data-privacy` does not delete the dashboard. It branched from coordinator before `d8c7627` added it.

**Goal:** one branch, `integration/demo`, where a single command runs a live 4-institution
Flower session on PathMNIST with our best CNN and recipe, activation noise, per-round
evaluation, MLflow tracking and the live dashboard. The tracking UI also shows the
model-improvement story (Steps 1–3).

**Scope:** integration only. No new model experiments beyond re-validating the recipe
in the Flower path. v2 and longer training stay deferred (`docs/accuracy_plan.md`, Phase 3–4).

## What the demo will show

1. **Setup:** `prepare` gives each institution its own PathMNIST shard of 22,499 images. Raw data never leaves it.
2. **Live session:** 4 institutions take turns. Each owner runs its Head, adds noise and sends activations through the relay; a different institution runs the shared Backbone. Gradients return and the Backbone checkpoint moves on.
3. **Live dashboard:** current owner and peer, Backbone version, per-turn loss and accuracy, activation change from noise, and per-round validation accuracy for every institution.
4. **MLflow:** this session's run next to the curated runs (baseline 0.791 → augmentation 0.873 → GroupNorm + recipe 0.883) and the noise comparison.
5. **Privacy:** the activation heatmap before and after noise, and the accuracy cost of each noise level.

## Conflicts and how to resolve them

| Area | `feature/data-privacy` | `split_learning` (ours) | Resolution |
| --- | --- | --- | --- |
| `SplitModelConfig` | adds `group_norm` | adds `batch_norm`, `backbone_final_pool`, `tail_dropout`, `group_norm`, `convs_per_block` | Union of fields. One `group_norm` field. |
| GroupNorm groups | `channels // 4` (4 channels per group); test asserts `[4, 8, 8]` | fixed 8 groups (`NORM_GROUPS`) | **Fixed 8 groups.** It is the setting behind our measured 0.883. Update their test to expect 8. **Member 3 to confirm.** |
| `conv_blocks`, `Head`, `Backbone` signatures | 3rd positional arg is `group_norm` | 3rd positional arg is `batch_norm` | Make every norm/pool option keyword-only (`*,`) so positional calls cannot silently swap meaning. `build_*` are the only callers. |
| Optimizer | `optimizer_factory(name, lr)` with `adam`/`sgd`; checkpoints record optimizer type | `experiments/pathmnist/recipe.adamw(lr, wd)` stamps `base_lr` | Keep theirs and extend it: add `adamw` and a `weight_decay` argument. Our recipe calls it. Keep the checkpoint type check. |
| `training.evaluate` | optional `perturb` | unchanged | Take theirs. |
| `Institution` | unchanged | optional `loss_fn` | Take ours. |
| `experiments/__init__.py` | docstring | empty | Take theirs. Its rule (experiments depend on everything, nothing depends on experiments) already fits `experiments/pathmnist/`. |
| Data loading | `institution_data`: stratified shards, inputs in [0, 1], per-turn batch stream from the round number | `experiments/pathmnist/data.py`: own download, random IID split, inputs in [-1, 1] | **Use theirs everywhere.** Delete our download and partition code, which duplicates theirs (global rules §3). Move `augment.py` into `institution_data/` and make it range-agnostic. Re-validate accuracy on [0, 1] inputs (see Verification). |
| MLflow | `tracking/` interface; store `~/.medweave/mlflow`; only `tracking/mlflow_tracker.py` imports mlflow | `run.py` and `compare.py` call mlflow directly; store `./mlflow.db` | One store: set `MEDWEAVE_TRACKING_DIR` to the repo root in the launcher, so both write to `./mlflow.db`. Port `run.py` to the `Tracker` interface. Drop `--workers` (it needs MLflow parent-run plumbing the interface does not have). `compare.py` already groups seeds by tag. |
| `README.md`, `split_learning/README.md`, `.gitignore`, engine tests | edited | edited | Merge by hand. Keep both sides' content. |

## Integration steps

Each step ends with the full test suite green. The owner reviews changes to their module (global rules §4).

### 1. Branch and merge (Member 2, about 2 h)

1. `git switch -c integration/demo origin/feature/data-privacy`
2. `git merge split_learning`. Resolve as in the table above. Engine tests from both sides must pass.
3. `git merge origin/coordinator` (brings `d8c7627`: dashboard and database). Expect conflicts in
   `coordinator/flower_app/session.py`, `server_app.py` and `tests/unit/coordinator/test_session.py`, where the
   dashboard's observer hooks in and Member 3 also changed the session. Keep Member 3's session
   (evaluation, tracking) and re-attach the observer calls. **Member 4 to review.**
4. Commit each merge separately so each can be reviewed and reverted.

### 2. Put the CNN into the Flower path (Member 2, about 1 h)

- `split_learning/path_cnn.py`: add `norm: Literal["batch", "group"] = "batch"` to `path_cnn_v1_config`.
  The default stays as specified; the demo uses `"group"`.
- `coordinator/flower_app/protocol.py`: `MODEL_CONFIG = path_cnn_v1_config(norm="group")`.
  `MODEL_CONFIG` is the single place the ClientApp, ServerApp and noise comparison build models from,
  so nothing else changes.
- Old institution checkpoints and `.medweave_local/` no longer fit and must be deleted (as when GroupNorm landed).
- The boundary tensor doubles: `[B, 32, 14, 14]` instead of `[B, 16, 14, 14]`, about 1.6 MB per
  64-image batch each way. That is fine on localhost; measure it before the multi-laptop deployment.

### 3. Bring the best recipe into the Flower path (Members 1 and 3 review, about 3 h)

| Recipe part | Change | Module |
| --- | --- | --- |
| AdamW + weight decay | run settings `optimizer = "adamw"`, `weight-decay = 1e-4` → `optimizer_factory` | `split_learning/config.py`, `client_app.py` |
| Cosine LR with warm-up | The coordinator computes one learning rate per turn from (turn index, total turns) and sends it in a new `TurnInfo.learning_rate` field. Owner and peer both apply it. **Apply it after `load_institution_checkpoint`**, because loading optimizer state restores the old learning rate. | `coordinator/flower_app/protocol.py`, `session.py`, `client_app.py`; schedule maths moved from `experiments/pathmnist/recipe.py` to `split_learning/schedule.py` |
| Augmentation | run setting `augment = true`. `training_loader` applies flips, rotations and colour jitter, seeded with `derive_seed(seed, institution, round)` so a turn stays a pure function of the round number (their decision record). | `institution_data/augment.py`, `batches.py` |
| Rounds | `num-rounds = 10` for the recorded reference session; 3 for the live demo | `pyproject.toml` / launcher |

The plan uses one learning rate per turn instead of per batch. That needs one field and no change to the
relay or `roles.py`, and with 40 turns in 10 rounds the steps are fine-grained. Step 5 checks that it
matches the per-batch schedule within the seed spread before we rely on it.

### 4. Recalibrate noise for the new CNN (Member 3, about 1 h of compute)

Their noise levels (low 0.25, high 1.25) were chosen relative to the placeholder's activation std
(about 0.69). Our Head output is larger and normalised differently:
- Measure the new activation std with `privacy.statistics`.
- Rescale low and high to the same *relative* change.
- Re-run `python -m experiments.noise_comparison` so `experiments/results/` matches the demo model.
- Pick the demo's default `noise-std` from that curve: the largest level that costs at most 2 accuracy points.

### 5. Windows launcher and demo script (Member 2, about 2 h)

- `pip install "flwr>=1.43"` (not installed on this laptop). Add `flwr` to a top-level `requirements.txt`.
- `coordinator/flower_app/run_local.ps1`: the same steps as `run_local.sh`, using `venv\Scripts`,
  `Start-Process` and the same `flwr ls` status check. It sets `MEDWEAVE_TRACKING_DIR` to the repo root.
  Keep `run_local.sh` for macOS and Linux.
- `scripts/demo.ps1` / `demo.sh`: prepare data → start the dashboard → run a 3-round session →
  open MLflow (`python -m tracking ui`).
- **Risk:** Flower's SuperLink/SuperNode on native Windows is untested here. Smoke-test a 1-round,
  5-batch session first. Fall back to WSL if it fails.

### 6. Migrate the experiment harness onto the shared modules (Member 2, about 2 h)

- `experiments/pathmnist/run.py` reads data through `institution_data`, noise through
  `privacy.GaussianActivationNoise` and tracking through `tracking.open_tracker`. Delete `data.py`'s
  download, partition and batching code; keep `metrics.py` and `compare.py`.
- Copy the four curated runs (Steps 1–3 and the centralized reference) into the shared store with the
  existing curation script, so the demo's MLflow shows the improvement story beside live sessions.

### 7. Documentation (all, about 1 h)

- `docs/DECISIONS.md`: entries for the group-count choice, the per-turn LR, augmentation determinism,
  the shared tracking store and the Windows launcher.
- `docs/shared_interfaces.md`: `TurnInfo.learning_rate`, the new run settings, `optimizer_factory("adamw", lr, weight_decay)`.
- Root `README.md`: a "Run the demo" section (one command per OS) with expected runtime and output.

## Verification

1. **Tests:** `python -m pytest tests` stays green after each step: about 218 tests from data-privacy,
   145 from ours and the dashboard/database tests, plus the integration test
   `tests/integration/test_data_and_noise_through_engine.py` with `MODEL_CONFIG` set to the new CNN.
2. **Recipe parity, in-process:** run our harness on `institution_data` shards with [0, 1] inputs and the
   per-turn LR, 10 rounds, 2 seeds. Accept if mean test accuracy is within 1 point of 0.883.
3. **Flower smoke test:** a 1-round, 5-batch session ends `finished:completed`. The dashboard shows all 4
   institutions, and MLflow gets a `medweave-sessions` run.
4. **Reference session:** a 10-round Flower session with the full recipe and the chosen noise level.
   Per-round validation accuracy should track the in-process run, i.e. land within the seed spread of it
   (their decision log shows Flower and in-process agree exactly when the maths matches). Record its
   duration; this becomes the pre-recorded run shown in the demo.
5. **Live demo rehearsal:** `demo.ps1` from a clean checkout and clean `.medweave_local/`, end to end, timed.

## Rough timeline

| Step | Work | Compute |
| --- | --- | --- |
| 1 Merge | 2 h | — |
| 2 CNN in Flower | 1 h | — |
| 3 Recipe in Flower | 3 h | — |
| 4 Noise recalibration | 0.5 h | about 1 h |
| 5 Launcher + smoke test | 2 h | — |
| 6 Harness migration | 2 h | about 1 h (parity check) |
| 7 Docs | 1 h | — |
| Reference session + rehearsal | 0.5 h | about 1 h |

That is about **1.5–2 working days**. Steps 4, 5 and 6 can run in parallel after Step 3.

## Decisions needed from the team

1. **GroupNorm group count:** fixed 8 (measured best with the demo CNN) or 4 channels per group
   (Member 3's default). Recommendation: fixed 8.
2. **Test-set ownership:** may the demo report a final test-split evaluation? Their sessions evaluate
   on `val` only. Recommendation: a run setting `final-evaluation-split = "test"`, used only in the last round.
3. **Demo noise level:** from the recalibrated curve (Step 4).
4. **Live rounds:** 3 rounds live (estimated 5–8 minutes on this laptop; measured in Step 5) plus the recorded 10-round session.
5. **Demo machine OS:** if the demo runs on macOS, Step 5's PowerShell launcher drops in priority.

## Risks

| Risk | Mitigation |
| --- | --- |
| Flower on native Windows | Smoke test first (Step 5); WSL fallback |
| Accuracy drops after switching to [0, 1] inputs, stratified shards or per-turn LR | Parity check (Verification 2) before touching Flower; if it fails, find which change caused it by switching them back one at a time |
| Merge of the dashboard into the changed session loop | Separate merge commit, reviewed by Member 4; dashboard tests must pass |
| Larger activations make turns slower | Measure turn time in the smoke test; `batches-per-turn` already gives a lever |
| Old checkpoints no longer match | `demo` script starts from a clean `.medweave_local/` |
