# Plan: maximise MedWeavePathCNN accuracy on PathMNIST

## Context

The first full run (MLflow run `7960d2e0`, commit `fed2e94`, 10 rounds, 349 s on CPU)
reached **mean test accuracy 0.775 / AUC 0.960**, against validation accuracy 0.902.
For reference, the MedMNIST v2 paper reports 0.907 / 0.983 for a centrally trained
ResNet-18 at 28×28. This plan explains where the missing accuracy goes and lays out
an ordered, measurable way to recover it.

**Target:** mean test accuracy ≥ 0.88 (stretch 0.90), AUC ≥ 0.98, spread between
institutions ≤ 3 points, validation swing between rounds ≤ 2 points.

**What approval triggers:** I save this document to `docs/accuracy_plan.md` and then
work through the phases below in order, stopping at each decision gate to report.
Phases 2b and 3 change checkpoint keys, so they ship as a new preset
(`path_cnn_v2_config`) and leave v1 intact.

## Current model and training recipe

| Part | Layers | Output | Params |
| --- | --- | --- | --- |
| Head | Conv 3→32, BN, ReLU, MaxPool | 32×14×14 | 960 |
| Backbone | Conv 32→64, BN, ReLU, MaxPool; Conv 64→128, BN, ReLU | 128×7×7 | 92,736 |
| Tail | AvgPool, Dropout 0.2, Linear 128→9 | 9 logits | 1,161 |

- One conv per stage, 3 convs total, receptive field 18×18 of the 28×28 patch.
- Recipe (`experiments/pathmnist/run.py`): Adam, constant lr 1e-3, batch 64, no
  weight decay, no augmentation, no LR schedule, final-round weights reported.
- Protocol: each turn is one full local epoch (~352 batches) for one institution;
  the Backbone optimizer is rebuilt at every handoff (`ComputePeer.load_backbone`).

## Evidence from the logged run

| | inst-1 | inst-2 | inst-3 | inst-4 |
| --- | --- | --- | --- | --- |
| Validation accuracy | 0.873 | 0.943 | 0.918 | 0.872 |
| Test accuracy | 0.783 | 0.828 | 0.828 | 0.662 |
| Recall ADI (adipose) | 0.74 | 0.94 | 0.97 | **0.44** |
| Recall MUC (mucus) | 0.90 | 0.80 | 0.78 | **0.28** |
| Recall DEB / MUS / STR | 0.34 / 0.44 / 0.28 | 0.47 / 0.55 / 0.40 | 0.32 / 0.48 / 0.47 | 0.51 / 0.69 / 0.19 |

Train accuracy in the last round was 0.943. Mean validation accuracy by round was
0.82, 0.82, 0.78, 0.91, 0.90, 0.88, 0.88, 0.90, 0.87, 0.90.

Three separate problems show up:

1. **Unstable evaluation.** Validation accuracy swings by up to 13 points between
   rounds while train loss falls smoothly, and institutions sharing one Backbone
   differ by 17 points on test. inst-4 sends 645 of 1,338 adipose patches to
   "muscle"; inst-3 gets 97% of them right. Adipose and mucus are easy,
   brightness-driven classes, so this is a calibration failure, not a capacity one.
2. **Weak generalisation to the test cohort.** The 12.7-point gap between
   validation and test is the PathMNIST domain shift (test images come from a
   different patient cohort, CRC-VAL-HE-7K). Nothing in the recipe counters it.
3. **Limited capacity on texture classes.** Debris, muscle and stroma are below
   0.70 recall for every institution. These need more depth and receptive field.

## Likely causes, ranked (hypotheses until Phase 0 confirms them)

| # | Cause | Why it fits the evidence | Fix |
| --- | --- | --- | --- |
| 1 | Shared BatchNorm running statistics. The Backbone's statistics are overwritten by whichever institution trained last, while four Heads that drift apart feed it. Training uses batch statistics, evaluation uses running ones. | Smooth train loss, erratic evaluation, errors on brightness-driven classes | GroupNorm, or recalibrate statistics before evaluating |
| 2 | Constant lr 1e-3 with no schedule and final-iterate reporting | Round-to-round validation swings | Cosine decay, select the round on validation |
| 3 | No augmentation or regularisation | Validation 0.90 vs test 0.775 | Flips, 90° rotations, colour jitter, weight decay, test-time augmentation |
| 4 | Long turns plus Backbone optimizer reset each handoff | Each Head/Tail trains once per round, then sits for three turns while the Backbone moves | Shorter turns, carry optimizer state |
| 5 | Shallow network | DEB/MUS/STR weak everywhere | Two convs per stage (v2) |

Without Phase 0 it is not possible to say how much of the 13-point gap each cause
owns, so fixes are ordered cheapest and most diagnostic first.

## Phase 0 — measurement harness (no model change)

All in `experiments/pathmnist/`; the engine is untouched.

- `--centralized`: train the same network unsplit on all training data with the
  same recipe. This is the ceiling for the architecture and separates "split
  learning costs X" from "the network tops out at Y". Reuse `build_head`,
  `build_backbone`, `build_tail` composed with `nn.Sequential`.
- Evaluation diagnostics per institution: eval-mode accuracy on its own training
  partition, and accuracy with BatchNorm statistics recalibrated on that
  institution's data. A large jump confirms cause 1.
- `--seeds 0,1,2`: one MLflow parent run with a child run per seed, logging mean
  and standard deviation. With a 17-point spread, single-run comparisons mean
  nothing.
- Select the reported round by mean validation accuracy and report test for that
  round only (reuse `save_backbone_checkpoint` and `save_institution_checkpoint`
  from `split_learning/checkpoint.py`). Test is never used for tuning.
- Log per-class recall as metrics and add a `phase` tag to every run.

**Gate:** report centralized vs split, and the size of the BatchNorm effect.
If recalibration alone recovers most of the instability, Phase 2b moves ahead of
Phase 1.

## Phase 1 — training recipe (checkpoint-compatible with v1)

Each item is a flag, added one at a time and kept only if mean validation
accuracy over 3 seeds improves beyond one standard deviation.

1. **Augmentation** (new `experiments/pathmnist/augment.py`, pure torch because
   torchvision is not installed): random horizontal/vertical flips and 90°
   rotations (histology has no canonical orientation), then brightness, contrast
   and saturation jitter. Applied per batch inside `batches()` for training only.
2. **AdamW with weight decay 1e-4** via the existing `optimizer_factory` hook.
3. **Cosine LR decay with a short warm-up** over the whole run. Institutions keep
   their optimizers, so the script sets `param_groups[...]["lr"]` per turn; the
   peer's factory reads the current value from a closure.
4. **More rounds:** 30 instead of 10 (about 18 minutes per v1 run on this CPU).
5. **Test-time augmentation:** average logits over the 8 flip/rotation views in
   `predict()`.
6. Dropout 0.2 → 0.3 and label smoothing 0.1 only if the gap to test persists.
   Label smoothing needs `Institution` to accept a `loss_fn` (the lower-level
   `tail_forward` already does), a small backward-compatible engine change.

**Gate:** expect the validation–test gap to shrink and swings to fall. Lock the
recipe before changing the model.

## Phase 2 — split-learning stabilisers

- **2a, experiment-only:**
  - `--turn-steps N`: hand over after N batches instead of a full epoch, with
    proportionally more rounds. Shorter turns track centralised SGD more closely
    on IID data. Cost: more checkpoint handoffs, which matters once the network
    is real, so log handoffs per run.
  - `--head-lr-scale {1, 0.1, 0}`: slow or freeze the Heads so the Backbone sees
    consistent inputs. Heads start identical (same seed), so frozen Heads stay
    identical while remaining local, which respects the spec.
  - Carry the Backbone optimizer state across handoffs, simulated through the
    optimizer factory. If it helps, propose adding optimizer state to the
    Backbone checkpoint in `shared_interfaces.md` (a team decision, not made here).
- **2b, engine config (my module):** add a normalisation choice to
  `SplitModelConfig` (`split_learning/config.py`, `layers.py`): keep
  `batch_norm` working, add GroupNorm (8 groups). GroupNorm has no running
  statistics, so nothing depends on who trained last, and it stays valid if the
  partitions later become non-IID. Existing defaults and v1 keys are unchanged;
  the existing key-stability tests guard that.

**Gate:** institution spread ≤ 3 points and validation swing ≤ 2 points.

## Phase 3 — capacity: MedWeavePathCNN-v2

Only after the recipe is fixed, so gains are attributable to the model.

- Add `convs_per_block` to `conv_blocks` (default 1 keeps today's layout):
  two 3×3 convs per stage at 32/64/128, about 0.29M parameters, receptive field
  covering the whole patch. Estimated 2–3× the step time.
- If still short of target, widen to 48/96/192 before considering residual
  blocks, which need a new block type.
- New preset `path_cnn_v2_config()` in `split_learning/path_cnn.py`; default
  split stays "stage 1 is the Head". No pretrained weights.

**Gate:** compare v2 against its own centralized ceiling from Phase 0 tooling.

## Phase 4 — lock and document

- Final configuration over 3 seeds, 30+ rounds; report mean ± std for accuracy,
  balanced accuracy, macro F1, AUC and per-class recall.
- Update `experiments/pathmnist/README.md` and `split_learning/README.md`
  (v2 section, layer table, checkpoint incompatibility with v1).
- One `--noise-std` sweep to show how accuracy degrades once activation noise is
  added, since the privacy layer will cost some of these gains.

## Constraints kept

- Heads and Tails stay institution-local and are never averaged or shared
  (SPEC §2.3). Averaging them SplitFed-style would likely help, but it needs a
  spec change and team agreement, so it is listed here only as an option.
- No change to gradient-exchange semantics, `Institution`, `ComputePeer` protocol
  or checkpoint format, except the opt-in items flagged above.
- CPU only, no pretrained weights, no new dependencies.
- Model selection on validation only.

## Files

| File | Change |
| --- | --- |
| `docs/accuracy_plan.md` | This document |
| `experiments/pathmnist/run.py` | Flags, seeds, centralized baseline, schedule, best-round selection, diagnostics |
| `experiments/pathmnist/augment.py` (new) | Flips, rotations, colour jitter, test-time views |
| `experiments/pathmnist/data.py` | Optional augmentation in `batches()` |
| `split_learning/config.py`, `layers.py`, `path_cnn.py` | Phases 2b and 3 only: normalisation choice, `convs_per_block`, v2 preset |
| `tests/unit/experiments/`, `tests/unit/split_learning/` | Tests for each new helper and config option |

## Verification

- `python -m pytest tests/unit -q` stays green (111 today), with new tests for:
  augmentation shape, dtype and label invariance; the 8 test-time views; the LR
  schedule values; default-config key stability; v2 shapes and split-point moves.
- Each phase ends with MLflow runs tagged by phase, compared in
  `mlflow ui --backend-store-uri sqlite:///mlflow.db` against run `7960d2e0`.
- Every reported number is a mean over 3 seeds with its standard deviation, and
  test metrics come from the validation-selected round.
