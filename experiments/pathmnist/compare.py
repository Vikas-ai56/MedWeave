"""Compare finished experiment runs: mean ± std over seeds, grouped by phase, label and mode.

    python -m experiments.pathmnist.compare
    python -m experiments.pathmnist.compare --phase 1 --metrics test/accuracy/mean,test/auc/mean

Groups the per-seed child runs (tag `seed`) so that seeds launched in separate
invocations still aggregate. Read-only.
"""

import argparse
import os
import statistics
from collections import defaultdict
from pathlib import Path

import mlflow

DEFAULT_METRICS = (
    "best_val/accuracy/mean",
    "test/accuracy/mean",
    "test_tta/accuracy/mean",
    "test/macro_f1/mean",
    "test/auc/mean",
    "test/accuracy/spread",
    "diag/recal_val_accuracy/mean",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
INSTITUTIONS = ("inst-1", "inst-2", "inst-3", "inst-4")


def spread(metrics: dict[str, float], key: str = "test/accuracy") -> float | None:
    """Max minus min across institutions, or None for centralized runs."""
    values = [metrics[f"{key}/{id_}"] for id_ in INSTITUTIONS if f"{key}/{id_}" in metrics]
    return max(values) - min(values) if values else None


def summarize(values: list[float]) -> str:
    if not values:
        return "-"
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.4f} ± {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="medweave-pathmnist")
    parser.add_argument("--phase", default=None, help="only this phase tag")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()
    metrics = args.metrics.split(",")

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
    )
    filters = ["attributes.status = 'FINISHED'", "tags.seed != ''"]
    if args.phase is not None:
        filters.append(f"tags.phase = '{args.phase}'")
    runs = mlflow.search_runs(
        experiment_names=[args.experiment], filter_string=" and ".join(filters), output_format="list"
    )

    groups: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for run in runs:
        tags = run.data.tags
        values = dict(run.data.metrics)
        if (value := spread(values)) is not None:
            values["test/accuracy/spread"] = value
        groups[(tags.get("phase", ""), tags.get("label", ""), tags.get("mode", ""))].append(values)

    header = ["phase", "label", "mode", "seeds", *metrics]
    rows = [
        [phase, label or "-", mode, str(len(results)), *(summarize([r[m] for r in results if m in r]) for m in metrics)]
        for (phase, label, mode), results in sorted(groups.items())
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


if __name__ == "__main__":
    main()
