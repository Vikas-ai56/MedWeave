# Decisions

## [2026-11-12] PathMNIST from the pinned Zenodo file, not the `medmnist` package
**Decision:** Download `pathmnist.npz` from Zenodo record 10519652 and verify MD5
`a8b06965200029087d5bd730944a56c1` (both from medmnist 3.0.2 metadata); read it with NumPy.
**Why:** The file and its counts are the agreed dataset; the reader needs only NumPy and torch.
**Rejected alternatives:** `pip install medmnist` (pulls pandas, scikit-learn, scikit-image, torchvision,
fire for one `np.load`).
**Risks / edge cases found:** Zenodo download took ~4.5 min here; it happens once per machine and before
Flower starts, never inside a turn. A wrong or partial file is rejected by MD5.

## [2026-11-12] Institutions are provisioned with their own shard before a session
**Decision:** `python -m institution_data prepare` (run by `run_local.sh`) writes one shard per
institution plus a manifest under `$MEDWEAVE_DATA_DIR` (default `~/.medweave/data`). The ClientApp opens
only its own shard. Partitioning is stratified round-robin over seeded per-class shuffles.
**Why:** A ClientApp only knows its own id, and in deployment each laptop holds only its own data;
partitioning inside the ClientApp would need the full roster. Round-robin gives exactly 22,499 each and
class counts within one.
**Rejected alternatives:** partitioning inside each ClientApp from `num-institutions`; storing data under
`.medweave_local/` (a state reset would re-download 205 MB).
**Risks / edge cases found:** A non-default `partition-seed` needs `prepare --seed N` first (clear error
otherwise). Session tests now need provisioned data; they use a 40-sample fixture in the official format.

## [2026-11-12] A turn's batches are a pure function of the round number
**Decision:** Training is one deterministic stream of full batches over seeded epochs; round `r` with `n`
batches per turn reads positions `[(r-1)·n, r·n)`.
**Why:** The coordinator fixes `num_batches` per turn and rejects other counts, and each ClientApp call
builds a fresh Institution, so no iterator position can survive between turns.
**Rejected alternatives:** one full epoch per turn (would force `batches-per-turn` to the partition size);
keeping iterator state in Flower's `context.state`.
**Risks / edge cases found:** A restarted session restarts at round 1 and so replays round-1 batches
(same as the coordinator's existing turn-order limitation).

## [2026-11-12] Activation noise: absolute std, training-only, seeded per institution and round
**Decision:** `noise_std` is an absolute Gaussian std in activation units, set by run setting `noise-std`
(default 0). Seeds come from `derive_seed(noise-seed, institution, round)`; evaluation is clean. Low and
high levels for the comparison are 0.1 and 0.5, chosen from an untrained-Head activation std of ~0.28.
**Why:** Matches the guide's recommendation and keeps utility measured consistently.
**Rejected alternatives:** a relative-to-activation scale; Opacus or formal DP (out of scope per team
decisions).
**Risks / edge cases found:** At 0.5, noisy-train accuracy (~0.44) is far above clean validation (~0.21),
a train/eval mismatch. Noise does nothing for gradient-based label leakage.

## [2026-11-12] Noise comparison runs in-process with Adam and validation-only evaluation
**Decision:** `experiments/noise_comparison.py` follows the coordinator's rotation in one process, uses
Adam lr 1e-3 per party, and scores each owner on the official validation split after its turn.
**Why:** With the engine's default SGD lr 0.01 every condition stayed at the majority-class rate (~15%),
which would make any noise comparison meaningless; Adam reaches ~56% in 3 rounds. Test ownership is
undecided, so test is not touched.
**Rejected alternatives:** running the comparison through Flower (slower, same maths; the relay test
already proves equivalence with in-process training).
**Risks / edge cases found:** The Flower path then still used the engine default optimizer (loss ~2.20
≈ ln 9 in a demo); resolved by "Adam for every party, chosen by run setting" below.

## [2026-11-12] Adam for every party, chosen by run setting; checkpoints record their optimizer
**Decision:** `optimizer` (`adam`) and `learning-rate` (0.001) run settings feed
`split_learning.config.optimizer_factory`, used for Head, Tail and Backbone. Institution checkpoints
store the optimizer type and refuse to load into a different one. The engine's own default stays SGD 0.01.
**Why:** Same architecture and data, 400 batches: SGD 0.01 → 16% validation, Adam 1e-3 → 54%. In the
Flower path one round now reaches ~0.5 mean validation accuracy instead of ~0.10.
**Rejected alternatives:** changing the engine default (would silently change every caller and test);
a key-set comparison of optimizer state (rejects the same optimizer saved by another torch version).
**Risks / edge cases found:** torch accepts SGD state into Adam and crashes at `step()` with
`KeyError: 'betas'`; the recorded type turns that into a clear error. Checkpoints saved before this change
carry no type and load unchecked. The peer's Adam state is reset every turn (engine design).

## [2026-11-12] Every round ends with local held-out evaluation by every institution
**Decision:** After each round the coordinator sends the latest Backbone with an `evaluate.held_out`
message; each institution scores its own Head and Tail on the public validation split (setting
`evaluation-split`, `none` to skip) and returns only loss, accuracy and sample count.
`run_session` returns `SessionResult(turns, evaluations)`.
**Why:** The coordinator cannot evaluate alone (Heads and Tails never leave institutions), and before
this there was no held-out accuracy at all. Every institution already receives the Backbone as a
compute peer, so nothing new is exposed; the evaluation images are public.
**Rejected alternatives:** sending Heads and Tails to the coordinator (breaks locality); evaluating only
in the offline experiment.
**Risks / edge cases found:** Test-set ownership is still undecided, so the default split is `val`.
An institution evaluated before its first turn scores its untrained Head and Tail.

## [2026-11-12] run_local.sh exits on the SuperLink's run status
**Decision:** Start the run with `flwr run --format json`, stream with `flwr log --stream`, then read
`flwr ls --run-id --format json` and exit 0 only for `finished:completed`.
**Why:** `flwr run --stream` exits 0 even when the ServerApp raises; the SuperLink's status is the
authoritative signal (checked: failing run → `finished:failed`, exit 1).
**Rejected alternatives:** matching error text in the streamed log.
**Risks / edge cases found:** The status can lag the end of the stream; the script polls it for up to 30 s.

## [2026-11-12] The owner's data is opened inside the turn
**Decision:** The ClientApp passes a generator that opens the shard on first use, so data errors are
raised inside `run_owner_turn`.
**Why:** Opened before the turn, a data error skipped the turn's "done" message and the compute peer
waited the full `turn-timeout` (300 s) before the turn was reported failed.
**Risks / edge cases found:** None found; a session test asserts the failure arrives in under 10 s.

## [2026-11-12] One shard pass per turn; current CNN kept
**Decision:** `batches-per-turn` = 351 (22,499 // 64). The split CNN is unchanged.
**Why:** A 5-batch and a 50-batch turn both took 6.2 s: the cost is a fixed per-turn overhead, and a full
pass takes ~12 s, far inside the 300 s timeout. Trained in one piece over the four shards, wider
variants (51k and 111k parameters vs 14.6k) reached 0.656 and 0.663 validation accuracy vs 0.643.
**Rejected alternatives:** the wider models (2 points for 3.5–7.6× the parameters; the widest also
doubles the transmitted activation size).
**Risks / edge cases found:** Changing `batch-size` without `batches-per-turn` no longer means one pass.

## [2026-11-12] Noisy evaluation is an option, clean is the default
**Decision:** `split_learning.training.evaluate` takes an optional `perturb`; the experiment's
`--evaluation clean|noisy` records the mode in every row. The Flower evaluation stays clean.
**Why:** The team has not decided; both numbers can now be produced and compared.
**Risks / edge cases found:** Noisy evaluation uses its own seed stream, so it never replays training noise.

## [2026-11-12] GroupNorm in the split CNN; BatchNorm rejected; 2 rounds by default
**Decision:** `SplitModelConfig.group_norm` adds GroupNorm after every conv (4 channels per group);
`MODEL_CONFIG` turns it on. `num-rounds` default is 2 (about 2 minutes on one laptop). Noise levels for
the comparison are rescaled to the new activation size (std ≈ 0.69): low 0.25, high 1.25.
**Why:** Same budget (2 rounds × 351 batches, in-process): plain CNN 0.654 mean validation accuracy,
BatchNorm 0.583, GroupNorm 0.762. Real 2-round Flower session with noise 0.25: 0.763.
**Rejected alternatives:** BatchNorm: the shared Backbone's running statistics follow whichever
institution trained last, so other institutions' Heads are normalised with the wrong statistics at
evaluation (per-institution accuracy 0.43–0.71). Shorter turns (4 rounds × 175): 0.634 vs 0.654, no gain.
The ResNet-style model, augmentation and learning-rate schedules are deferred (longer training needed).
**Risks / edge cases found:** Checkpoints from the previous model no longer load; delete
`.medweave_local/` (and any old institution checkpoints) after pulling this change.

## [2026-11-12] MLflow tracking behind a small interface, coordinator-side only
**Decision:** New `tracking/` package: `Tracker` interface, `NullTracker`, `MlflowTracker` (only importer
of `mlflow`). Flower sessions (`tracking` setting, default `mlflow`) and the noise comparison (`--tracking`)
each become MLflow runs in a local SQLite store under `$MEDWEAVE_TRACKING_DIR` (default `~/.medweave/mlflow`).
**Why:** Runs were compared by reading logs and CSVs. The coordinator already receives every turn's metrics
and every round's evaluation, so it can record them without institutions changing or sending anything new.
**Rejected alternatives:** MLflow calls inside `session.py` or the institutions (hard dependency, more
data paths); a tracking server (not needed for one laptop; the guide puts server/UI work out of scope).
**Risks / edge cases found:** The ServerApp's working directory is Flower's app folder, so the store and
artifact locations are absolute. Real runs match untracked runs exactly (0.762/0.766/0.564; 0.763).
MLflow 3.18 on Python 3.14 logs a SQLAlchemy deprecation warning from its own code.

## [2026-11-13] GroupNorm uses a fixed 8 groups
**Decision:** `NORM_GROUPS = 8` for every GroupNorm layer (Member 2's setting), replacing Member 3's
`channels // 4`. `conv_blocks`, `Head` and `Backbone` take every norm/pool option keyword-only.
**Why:** It is the setting behind the measured 0.883 test accuracy (docs/integration_plan.md).
Keyword-only options stop a positional call from silently swapping `batch_norm` and `group_norm`,
which the two branches had in different positions.
**Rejected alternatives:** 4 channels per group.
**Risks / edge cases found:** Channel widths must be divisible by 8 (checked in `SplitModelConfig`).
Member 3 to confirm.

## [2026-11-13] MedWeavePathCNN-v1 with GroupNorm is the system model
**Decision:** `MODEL_CONFIG = path_cnn_v1_config(norm="group")`. Boundary tensor `[B, 32, 14, 14]`.
**Why:** Best measured model; GroupNorm keeps no running statistics, so the shared Backbone does not
carry the last institution's BatchNorm statistics.
**Risks / edge cases found:** Old Backbone and institution checkpoints no longer load; start from a
clean state directory. Each batch moves about 1.6 MB each way.

## [2026-11-13] One learning rate per turn, computed by the coordinator
**Decision:** The coordinator turns `learning-rate`, `lr-schedule` (`cosine`) and `lr-warmup` (0.05)
into one rate per turn and sends it in `TurnInfo.learning_rate`. Owner and peer both train at it; the
owner sets it after loading its checkpoint. The schedule maths is `split_learning/schedule.py`.
**Why:** One new field, no change to the relay or the per-batch protocol, and the coordinator already
knows the turn index and the total.
**Rejected alternatives:** a per-batch schedule (needs each party to count batches across turns).
**Risks / edge cases found:** Loading an institution checkpoint restores the previous turn's rate, so
it must be set afterwards; a test covers it. With few turns the warm-up is a single turn.

## [2026-11-13] Augmentation is seeded per institution and round
**Decision:** `augment = true`: flips, 90° rotations and colour jitter 0.2 (`institution_data/augment.py`),
with a generator seeded by `derive_seed(partition-seed, "augment", institution, round)`.
**Why:** It keeps a turn a pure function of the round number, like the batch order.
**Risks / edge cases found:** Moved from `experiments/pathmnist/augment.py`, which now wraps it for its
[-1, 1] inputs; the system's inputs are [0, 1].

## [2026-11-13] Demo noise 0.15 from the recalibrated curve
**Decision:** `noise-std = 0.15`, the largest level costing at most 2 accuracy points in a 3-round sweep
with the demo model and recipe (`experiments/results/noise_comparison_pathcnn_*`).
**Why:** The rule in docs/integration_plan.md, step 4. With the new model 0.25 costs 3.5 points.
**Risks / edge cases found:** One seed; run-to-run spread is about ±1 point.

## [2026-11-13] The last round also reports test accuracy
**Decision:** `final-evaluation-split = "test"`: after the last round every institution scores the
official test split once. Per-round evaluation stays on validation.
**Why:** Team decision (2026-11-13) so the demo can report the benchmark number.
**Risks / edge cases found:** Test is never used to choose anything; it is scored once at the end.

## [2026-11-13] TLS everywhere, with a private CA per deployment
**Decision:** `make_certs.sh` creates a CA and the coordinator's certificate. The SuperLink and the relay
serve TLS; SuperNodes, the `flwr` CLI and relay clients verify with the CA. `run_local.sh` and Docker
never use `--insecure`. TLS file paths come from `MEDWEAVE_TLS_CERT`, `MEDWEAVE_TLS_KEY`, `MEDWEAVE_TLS_CA`.
**Why:** SPEC §5.4 requires encrypted transport before the multi-laptop deployment.
**Rejected alternatives:** mutual TLS (institution certificates); not needed to encrypt the traffic.
**Risks / edge cases found:** Python 3.14 requires key-identifier extensions in the certificates. The
certificate must name the address institutions use (LAN IP or hostname), passed to `make_certs.sh`.

## [2026-11-13] Docker Compose deployment with one image
**Decision:** One image for every process; Compose runs setup, the coordinator, 4 institutions with
separate state volumes and the dashboard; `docker compose run --rm session` starts a run.
**Why:** One build to maintain, and the same commands as a real deployment.
**Risks / edge cases found:** The data volume is shared read-only in Compose. The dashboard's state
volume must be writable because SQLite WAL readers need the `-shm` file.
