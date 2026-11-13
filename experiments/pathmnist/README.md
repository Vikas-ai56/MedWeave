# PathMNIST experiment

End-to-end run of MedWeavePathCNN-v1 with sequential U-shaped split learning on
PathMNIST. All four institutions are simulated in one process, and every metric
is logged to MLflow.

This is a local simulation harness. The real coordinator, Flower, the privacy
layer and the shared dataset module are owned by other members and are not
used here.

## What a run does

1. Downloads `pathmnist.npz` (205 MB, official MedMNIST release on Zenodo) to
   `data/` if it is missing, and checks its MD5.
2. Splits the 89,996 training images into four disjoint IID partitions, one per
   institution (seeded).
3. For each round, runs four sequential turns. Institution *i* trains one local
   epoch on its partition while institution *i+1* runs the Backbone. The
   Backbone checkpoint is handed from each turn to the next.
4. After every round, scores each institution's Head + current Backbone + Tail
   on the validation split (10,004 images), and keeps checkpoints of the round
   with the best mean validation accuracy.
5. Restores that best round and scores the test split (7,180 images) once. The
   test split is never used to choose anything.
6. Runs BatchNorm diagnostics on the selected model (see below).
7. Logs the metrics, a `test_metrics.json` with confusion matrices, and the
   best Backbone checkpoint.

With `--centralized`, the same network is trained unsplit on the whole training
set instead. One round is one pass over the data in both modes, so the two see
the same samples. The centralized result is the ceiling for the architecture,
and the difference between the two modes is the cost of split learning.

With every option at its default, a run is the original baseline: v1,
BatchNorm, Adam (AdamW with no weight decay is the same optimizer) at a
constant 1e-3, no augmentation and no activation noise.

## Running

```bash
pip install -r experiments/requirements.txt
python -m experiments.pathmnist.run --rounds 10 --seeds 0,1,2 --workers 3
python -m experiments.pathmnist.run --rounds 10 --seeds 0,1,2 --workers 3 --centralized
python -m experiments.pathmnist.compare --phase 1         # mean ± std per configuration
mlflow ui --backend-store-uri sqlite:///mlflow.db         # http://127.0.0.1:5000
```

Run from the repository root. Options and their defaults:

| Group | Option | Default | Meaning |
| --- | --- | --- | --- |
| Schedule | `--rounds`, `--local-epochs`, `--batch-size` | 3, 1, 64 | One round = one pass over the training data |
| Optimizer | `--lr`, `--weight-decay` | 1e-3, 0 | AdamW for every component |
| | `--schedule`, `--warmup` | `constant`, 0.05 | `cosine`: linear warm-up over that fraction of steps, then cosine decay to 0, applied per batch |
| Regularisation | `--augment`, `--jitter` | off, 0.2 | Random flips and 90° rotations, then brightness, contrast and saturation jitter |
| | `--dropout`, `--label-smoothing` | 0.2, 0 | Tail dropout; label smoothing on the training loss |
| Evaluation | `--tta` | off | Also report `test_tta/*`: logits averaged over the 8 orientations |
| Model | `--model`, `--norm`, `--head-stages` | `v1`, `batch`, 1 | `v2` has two convs per stage; `group` uses GroupNorm |
| Split only | `--turn-steps` | 0 | Hand over after N batches instead of a full epoch |
| | `--head-lr-scale` | 1.0 | Head learning rate multiplier; 0 freezes the Heads' weights |
| | `--carry-backbone-optimizer` | off | Pass Backbone optimizer state from peer to peer |
| | `--noise-std` | 0 | Gaussian noise on outgoing activations, in training and evaluation |
| Runs | `--seeds`, `--workers`, `--threads` | `0`, 1, 0 | Comma-separated seeds; run them in parallel processes; torch threads each |
| Tracking | `--centralized`, `--phase`, `--label`, `--experiment`, `--data-dir` | | Mode, tags, MLflow experiment and data folder |

Each invocation is one parent MLflow run with a child run per seed. The parent
logs every final metric twice: as the mean over seeds under its own key, and as
the sample standard deviation under `std/{key}`. Runs are tagged with `mode`
(`split` or `centralized`), `phase`, `label` and `seed`. `compare.py` groups
child runs by phase, label and mode, so seeds started in separate invocations
still aggregate. It also reports `test/accuracy/spread`, the gap between the
best and worst institution.

**Memory.** Each trainer process holds the dataset and its libraries, about
1–1.3 GB. On a 16 GB machine, keep at most about 6 trainers running at once.

Runs go to `mlflow.db` in the repository root. Artifacts go to `mlruns/`,
relative to the directory you run from. Set `MLFLOW_TRACKING_URI` to log to a
different MLflow server instead. `data/`, `mlflow.db` and `mlruns/` are
gitignored.

## What is logged

| Key | Step | Meaning |
| --- | --- | --- |
| `train/loss`, `train/accuracy` (and `/inst-N`) | Backbone version (one per turn; centralized logs at round × 4) | Mean over the pass, in train mode |
| `val/{metric}/{id}`, `val/{metric}/mean` | round | Validation scores using the latest Backbone |
| `best_round`, `best_val/{metric}/mean` | final | The round selected on validation, and its scores |
| `test/{metric}/{id}`, `test/{metric}/mean` | final | Test scores for the selected round |
| `diag/train_eval_accuracy/{id}` | final | Eval-mode accuracy on the institution's own training partition |
| `diag/recal_{val,test}_accuracy/{id}` | final | Split only: accuracy after recomputing Head and Backbone BatchNorm statistics from that institution's partition |
| `test_tta/{metric}/{id}` | final | With `--tta`: test scores with logits averaged over 8 orientations |
| `handoffs` | final | Split only: Backbone checkpoint handoffs in the run |
| params | | Run config, seed, split shapes, parameter counts, partition sizes |
| artifacts | | `test_metrics.json` (metrics and confusion matrix per institution), `test_tta_metrics.json` with `--tta`, `backbone_best.pt` (split only) |

`{id}` is `inst-1` to `inst-4`, or `central` in centralized mode. Metrics are
`loss` (cross-entropy), `accuracy`, `balanced_accuracy`, `macro_f1`, `auc`
(macro one-vs-rest, the MedMNIST benchmark metric) and `recall_{class}` for
each of ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR and TUM.

**BatchNorm diagnostics.** During training, BatchNorm normalises with the
current batch's statistics. In eval mode it uses running averages, and in the
Backbone those are overwritten by whichever institution trained last. If
recomputing the statistics from an institution's own data raises its accuracy
a lot, the shared statistics, not the learned weights, are what cost accuracy.
Recalibration runs on copies, so the reported test metrics are unaffected.

## Baseline result

Run `7960d2e0` used commit `fed2e94` with the defaults except `--rounds 10`:
Adam at 1e-3, batch size 64, no noise, no augmentation, and the default split.
It took 349 s on CPU.

| | Validation accuracy | Test accuracy | Test balanced accuracy | Test macro F1 | Test AUC |
| --- | --- | --- | --- | --- | --- |
| inst-1 | 0.873 | 0.783 | 0.711 | 0.707 | 0.956 |
| inst-2 | 0.943 | 0.828 | 0.760 | 0.762 | 0.971 |
| inst-3 | 0.918 | 0.828 | 0.752 | 0.753 | 0.960 |
| inst-4 | 0.872 | 0.662 | 0.649 | 0.613 | 0.952 |
| **mean** | **0.902** | **0.775** | **0.718** | **0.709** | **0.960** |

Train loss fell from 0.80 to 0.16. The gap between validation and test is
expected for PathMNIST, because its test split comes from a different source
(CRC-VAL-HE-7K) than train and validation (NCT-CRC-HE-100K). For reference,
the MedMNIST v2 paper reports test accuracy 0.907 and AUC 0.983 for a
centralised ResNet-18 at 28×28.

Two things in this run are worth following up:

- **Validation accuracy swings between rounds.** The mean went 0.82, 0.82,
  0.78, 0.91, 0.90, 0.88, 0.88, 0.90, 0.87, 0.90, even though train loss fell
  every round.
- **Institutions differ a lot on the test set.** inst-4 trained last against
  the final Backbone, yet has the lowest test score. Its recall is weak on
  classes 0, 2, 4 and 7. So the spread does not come from the Backbone
  drifting after earlier turns. Each institution's Head and Tail generalise
  differently.

## Reading per-institution scores

Every institution's score uses its own Head and Tail with the shared Backbone
from the selected round. Earlier institutions last trained against an older
Backbone, so a gap between institutions can show how far the shared Backbone
drifted after their turn.
