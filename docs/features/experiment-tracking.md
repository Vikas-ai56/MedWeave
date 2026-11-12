# Experiment tracking (MLflow)

Every noise comparison and every Flower session is recorded in MLflow, so runs can be compared side by
side instead of by opening logs and CSVs.

## Use it

```bash
source venv/bin/activate
pip install mlflow                                  # once
coordinator/flower_app/run_local.sh --run-config "noise-std=0.25"
python -m experiments.noise_comparison --noise-std 0 0.25 1.25 --rounds 2
python -m tracking ui                               # then open http://127.0.0.1:5000
```

Turn it off with `--run-config "tracking=none"` (sessions) or `--tracking none` (experiments); then
MLflow does not need to be installed.

## What is recorded

| Experiment | Run | Params | Metrics (step) | Artifacts |
|---|---|---|---|---|
| `medweave-sessions` | one per Flower session, named `session-<flower run id>` | every run setting, the model config | `train_loss`, `train_accuracy`, `train_accuracy/<inst>`, `activation_change_pct`, `turn_seconds` (Backbone version); `val_accuracy/<inst>`, `val_loss/<inst>`, `val_accuracy_mean` (round) | final `backbone_vN.pt` |
| `medweave-noise-comparison` | one parent per command, one nested child per `noise_std` | all settings, model config (parent); `noise_std`, `evaluation` (child) | child: `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `val_accuracy/<inst>`, `activation_change_pct` (Backbone version), `final_val_accuracy_mean`, `mean_activation_change_pct`; parent: `final_val_accuracy/noise_<x>` | CSV, settings JSON, partition manifest |

Tags `git_commit` and `git_dirty` identify the code. A session or experiment that raises is marked FAILED.

**Never recorded:** images, labels, activations, gradients or per-sample predictions. Only the
coordinator logs sessions, and only the numbers it already receives; institutions do not import MLflow.

## Where and how

- Store: `sqlite:///<tracking dir>/mlflow.db`, artifacts in `<tracking dir>/artifacts/<experiment>/`;
  tracking dir `$MEDWEAVE_TRACKING_DIR`, default `~/.medweave/mlflow` (survives `rm -rf .medweave_local`).
- Code: `tracking/` (`Tracker` interface, `NullTracker`, `MlflowTracker`, `open_tracker`). Only
  `tracking/mlflow_tracker.py` imports MLflow. `coordinator/flower_app/session.py` logs through the
  interface; `server_app.py` opens the run; `experiments/noise_comparison.py` does the same.
- Tests use a recording tracker for the session and a temporary MLflow store for the backend.
- On the multi-laptop setup only the coordinator machine needs MLflow.
