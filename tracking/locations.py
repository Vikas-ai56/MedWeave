"""Where tracked runs live on this machine, which backend to open, and the source revision of a run.

Store: `sqlite:///<tracking dir>/mlflow.db`, artifacts in `<tracking dir>/artifacts/<experiment>`, with the
tracking dir from `$MEDWEAVE_TRACKING_DIR` (default `~/.medweave/mlflow`), outside the session state so
`rm -rf .medweave_local` keeps experiment history. `open_tracker("none" | "mlflow", experiment)` imports
MLflow only when asked for, and fails with an install hint if it is missing. `source_tags()` records the
git commit and whether the working tree had uncommitted changes.
"""

import os
import subprocess
from pathlib import Path

from tracking.tracker import NullTracker, Tracker

TRACKING_KINDS = ("mlflow", "none")
REPO_ROOT = Path(__file__).resolve().parent.parent


def tracking_dir() -> Path:
    return Path(os.environ.get("MEDWEAVE_TRACKING_DIR", Path.home() / ".medweave" / "mlflow"))


def tracking_uri() -> str:
    return f"sqlite:///{tracking_dir() / 'mlflow.db'}"


def open_tracker(kind: str, experiment: str) -> Tracker:
    if kind == "none":
        return NullTracker()
    if kind != "mlflow":
        raise ValueError(f"tracking must be one of {TRACKING_KINDS}, got {kind!r}")
    try:
        from tracking.mlflow_tracker import MlflowTracker
    except ImportError as error:
        raise RuntimeError("tracking is 'mlflow' but mlflow is not installed: pip install mlflow, or use tracking 'none'") from error
    directory = tracking_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return MlflowTracker(tracking_uri(), experiment, directory / "artifacts")


def source_tags() -> dict[str, str]:
    def git(*arguments: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *arguments], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": str(bool(git("status", "--porcelain"))).lower()}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}
