# Institutional data and activation noise (Member 3)

Member 3 owns two things: **data in** (PathMNIST → four disjoint stratified institutional shards →
per-turn DataLoaders, labels stay local) and **activations out** (Head activations → Gaussian noise →
perturbation statistics → the noisy tensor handed to the coordinator relay). Packages:
`institution_data/` and `privacy/`. The experiment runner is in `experiments/`.

## Quick start

```bash
source venv/bin/activate
python -m institution_data prepare                       # download (205 MB, MD5-checked) + shards + proof table
coordinator/flower_app/run_local.sh --run-config "noise-std=0.1"
python -m experiments.noise_comparison --inspect         # activation magnitudes, to pick noise_std
python -m experiments.noise_comparison --noise-std 0 0.1 0.5 --rounds 3 --batches-per-turn 100
```

`run_local.sh` runs `prepare` itself before starting Flower, so the first command is only needed to
pre-download or to see the partition table.

## Dataset

| Item | Value |
|---|---|
| Source | Zenodo record 10519652 `pathmnist.npz`, MD5 `a8b06965200029087d5bd730944a56c1` (medmnist 3.0.2 metadata), CC BY 4.0 |
| Official splits | train 89,996 · validation 10,004 · test 7,180 (test is CRC-VAL-HE-7K, a different clinical centre, so test accuracy will sit below validation) |
| Data root | `$MEDWEAVE_DATA_DIR`, default `~/.medweave/data` (outside `.medweave_local/`, so a state reset does not re-download) |
| Layout | `pathmnist/pathmnist.npz`, `pathmnist/partition-seed<N>/manifest.json`, `pathmnist/partition-seed<N>/inst-K.npz` |

**Partitioning** (`institution_data/partition.py`): shuffle each class with the seed, lay the classes
end to end, deal round-robin to the institutions. Result for seed 0: 22,499 samples each, every class
split to within one sample, disjoint, every training index used exactly once. Validation and test are
never partitioned and never used for training.

**Shards**: `prepare` writes one `inst-K.npz` per institution (its images, labels and official training
indices) plus `manifest.json` (dataset, source MD5, seed, sizes, per-class counts, SHA-256 of each index
list). A ClientApp opens only its own shard; a shard whose size or index digest disagrees with the
manifest is refused. This mirrors the multi-laptop deployment, where each laptop holds only its shard.

**Batches** (`institution_data/batches.py`): images float32 `[B, 3, 28, 28]` in [0, 1] (HWC → CHW,
/255, no normalisation or augmentation), labels int64 `[B]` in 0..8. Training is a deterministic stream
of full batches over seeded epochs; a turn of `n` batches in round `r` reads stream positions
`[(r-1)·n, r·n)`. A turn therefore needs no state from earlier turns, and the ClientApp rebuilds its
loader from `TurnInfo.round_number`. Each epoch's trailing partial batch is dropped; later epochs use a
different order, so those samples are still reached.

## Activation noise

`A' = A + noise_std · N(0, 1)` (`privacy/noise.py`). `noise_std` is an absolute standard deviation in
activation units. The output keeps shape, dtype and device, the input is never modified,
`noise_std = 0` is an exact copy, and invalid scales or non-finite activations are rejected. Gradients
pass through unchanged: the engine detaches the Head output before the hook and applies the returned
gradient straight-through, which is exact for additive noise. Do not change that detach.

In plain terms: the noise is never removed. The compute peer works on the noisy A', and so do the
Backbone output, the prediction and the loss. The model learns from a slightly fuzzy signal. The gradient
the peer returns for A' is also the exact gradient for A, because A' = A + (a fixed number), so the Head
applies it directly. Labels never leave the institution, but the gradient the owner sends to the peer
is computed from them; that is the leakage channel noise does not cover.

`GaussianActivationNoise(noise_std, seed)` is the `perturb` hook (`Callable[[Tensor], Tensor]`) given to
`run_owner_turn` / `train_batches`. Each (institution, round) gets its own reproducible stream from
`derive_seed(noise-seed, institution_id, round_number)`. Noise is **training-only** by default:
evaluation uses clean activations. `split_learning.training.evaluate(..., perturb=...)` and the
experiment's `--evaluation noisy` produce the noisy-evaluation number when wanted.

**Statistics** (`privacy/statistics.py`): original, noisy and noise L2 norms, mean and std before and
after, and relative change `||A' − A|| / (||A|| + 1e-12) × 100`. `activation_heatmaps` returns one
channel of one sample (original, noisy, difference) for the dashboard. Only these small derived values
leave the institution; whole activation tensors are never stored for the dashboard. They measure how
much the representation changed, not whether privacy is achieved.

## Run settings (`coordinator/flower_app/pyproject.toml`)

| Setting | Default | Meaning |
|---|---|---|
| `partition-seed` | 0 | Which provisioned partition and batch order. `run_local.sh` prepares seed 0; for another seed run `python -m institution_data prepare --seed N` first |
| `batch-size` | 64 | Training batch size per institution |
| `noise-std` | 0.15 | Gaussian noise std on Head activations (0 = baseline); chosen from the 2026-11-13 sweep below |
| `noise-seed` | 0 | Base seed for the noise streams |

Each owner turn reports `noise_std` and `activation_change_pct` in its MetricRecord. The coordinator
logs them and keeps them in `TurnResult`.

## Experiment result (2026-11-13, MedWeavePathCNN-v1 with GroupNorm)

The system model changed to Member 2's MedWeavePathCNN-v1 with GroupNorm and the full recipe (AdamW with
weight decay 1e-4, per-turn cosine learning rate, augmentation), so the levels were measured again.
`experiments.noise_comparison` now trains with that recipe by default.

```bash
python -m experiments.noise_comparison --inspect      # untrained Head activation std 0.647 (45% zeros)
python -m experiments.noise_comparison --noise-std 0 0.25 0.5 0.75 1.0 1.25 --rounds 3 \
    --output experiments/results/noise_comparison_pathcnn_clean_eval.csv
python -m experiments.noise_comparison --noise-std 0.05 0.1 0.15 --rounds 3 \
    --output experiments/results/noise_comparison_pathcnn_low_clean_eval.csv
```

3 rounds, 351 batches per turn, partition and noise seed 0, clean evaluation on the validation split:

| noise_std | Final-round mean validation accuracy | Cost (points) | Mean activation change |
|---|---|---|---|
| 0 | 0.817 | – | 0% |
| 0.05 | 0.813 | 0.4 | 6% |
| 0.10 | 0.797 | 2.0 | 12% |
| **0.15** | **0.802** | **1.5** | **17%** |
| 0.25 | 0.782 | 3.5 | 27% |
| 0.5 | 0.760 | 5.7 | 52% |
| 0.75 | 0.670 | 14.7 | 76% |
| 1.0 | 0.559 | 25.8 | 100% |
| 1.25 | 0.443 | 37.4 | 122% |

- **Noise now costs accuracy sooner.** With the earlier placeholder CNN, 0.25 was free; with this
  model it costs 3.5 points.
- **The demo uses 0.15:** the largest level costing at most 2 points, as docs/integration_plan.md asks.
- **Spread:** only one seed was run. 0.10 scoring below 0.15 shows run-to-run variation of about ±1 point.
- **Real Flower session** (3 rounds, noise 0.15, TLS): validation 0.704, 0.777, 0.798 per round, which
  matches the in-process 0.802. Final test accuracy was 0.831 (institutions 0.825–0.835).

## Experiment result (2026-11-12, placeholder GroupNorm model)

```bash
python -m experiments.noise_comparison --inspect
python -m experiments.noise_comparison --noise-std 0 0.25 1.25 --rounds 2 --batches-per-turn 351 --evaluation clean \
    --output experiments/results/noise_comparison_clean_eval.csv
python -m experiments.noise_comparison --noise-std 0 0.25 1.25 --rounds 2 --batches-per-turn 351 --evaluation noisy \
    --output experiments/results/noise_comparison_noisy_eval.csv
```

Adam lr 1e-3, batch 64, one full shard pass per turn, 2 rounds (8 turns), partition seed 0, noise seed
0, official validation split. The untrained GroupNorm Head's activations have std ≈ 0.69 (40% zeros),
so 0.25 (≈0.36× std) is "low" and 1.25 (≈1.8× std) is "high", the same ratios as the earlier plain-CNN
run (0.1 / 0.5 against std 0.28). Each CSV has one row per turn; the `.settings.json` beside it records
every setting and the model config.

| noise_std | Clean evaluation | Noisy evaluation | Mean activation change |
|---|---|---|---|
| 0 | 0.762 | 0.762 | 0% |
| 0.25 | 0.766 | 0.758 | 26% |
| 1.25 | 0.564 | 0.661 | 122% |

(Final-round mean validation accuracy over the four institutions.) These runs are also recorded in
MLflow, experiment `medweave-noise-comparison` (`docs/features/experiment-tracking.md`).

- Low noise costs nothing measurable: a 26% change to every outgoing activation, same accuracy.
- High noise costs 10–20 points, less when evaluation also uses noise, because the model adapted to it.
- Earlier plain-CNN run (no GroupNorm, 3 rounds): 0.665 / 0.624 / 0.172 clean. GroupNorm makes the model
  both more accurate and far more tolerant of noise.
- The real 2-round Flower session with noise 0.25 reached 0.763, matching the in-process number.

## What this does not do

- **Not differential privacy.** No clipping, accountant or guarantee; `noise_std` is not an epsilon.
- **No protection for the gradient channel.** The gradient returned to the peer is derived from the
  labels. The team's audit recovered 320/320 labels from gradients in a tested configuration. Activation
  noise does not change that, and it remains an unresolved system-level issue.
- No reconstruction-attack evaluation.

## Open decisions (not guessed)

| Question | Current state |
|---|---|
| Who owns the test set | Decided 2026-11-13: every round is scored on `val`; after the last round each institution also scores `test` once (`final-evaluation-split = "test"`). |
| Noise during evaluation | Clean by default; noisy evaluation is available in the engine and the experiment, and both results are below. |
| Visualisation sharing | Only derived statistics and single-channel heatmaps are produced; nothing is sent to the dashboard yet. |
