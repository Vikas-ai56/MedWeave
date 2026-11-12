"""Experiment tracking for MedWeave: one small interface, MLflow behind it, nothing else depends on MLflow.

Public interface:
- `open_tracker("mlflow" | "none", experiment)` returns a `Tracker`; `NullTracker` is the no-op.
- `Tracker.run(name, params, tags, nested)`, `.log_metrics(metrics, step)`, `.log_artifact(path)`.
- `source_tags()`: git commit and dirty flag for a run; `tracking_uri()`: this machine's store.
- `python -m tracking ui` opens the MLflow UI on that store (http://127.0.0.1:5000).

Only aggregate numbers, settings and small files are tracked; never images, labels, activations or
gradients. Used by the coordinator's ServerApp (experiment `medweave-sessions`) and the noise comparison
(experiment `medweave-noise-comparison`); see docs/features/experiment-tracking.md.
"""

from tracking.locations import TRACKING_KINDS, open_tracker, source_tags, tracking_dir, tracking_uri
from tracking.tracker import NullTracker, Scalar, Tracker

__all__ = [
    "TRACKING_KINDS",
    "NullTracker",
    "Scalar",
    "Tracker",
    "open_tracker",
    "source_tags",
    "tracking_dir",
    "tracking_uri",
]
