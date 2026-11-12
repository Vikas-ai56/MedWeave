"""Activation-noise comparison on the provisioned PathMNIST shards, run in one process without Flower.

`python -m experiments.noise_comparison --inspect` prints Head activation statistics (one batch per
institution, untrained Head) so noise_std values can be chosen relative to activation magnitude.

`python -m experiments.noise_comparison --noise-std 0 <low> <high>` runs one condition per value. Every
condition starts from the same initial weights (SplitModelConfig seed), the same shards and the same
batch order, and follows the coordinator's rotation (build_schedule: each institution trains
`--batches-per-turn` batches per round through the shared Backbone, computed by the next institution).
Each party has its own optimizer (default Adam, lr 1e-3: the engine's default SGD lr 0.01 stays at
the majority-class rate on PathMNIST for this many batches). After each turn the data owner's own Head and Tail are evaluated with the new Backbone on the official
validation split, with clean activations by default or, with `--evaluation noisy`, with the same noise
level as training (its own seed stream); the mode is written into every row. Test-split evaluation is deliberately
absent until the team decides who owns the test set. Writes one CSV row per turn and a JSON file with the
exact settings next to it. With `--tracking mlflow` (default) the command is one MLflow run in
experiment `medweave-noise-comparison`, with one nested run per noise level: per-turn metrics at step =
Backbone version, final summaries, and the CSV, settings JSON and partition manifest as artifacts.
`--tracking none` needs no MLflow.
"""

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from coordinator.flower_app.protocol import MODEL_CONFIG
from coordinator.scheduling.schedule import build_schedule
from institution_data import (
    DEFAULT_INSTITUTION_IDS,
    ImageLabelDataset,
    PartitionManifest,
    default_data_dir,
    evaluation_loader,
    open_evaluation,
    open_institution_training,
    training_loader,
)
from institution_data.provision import MANIFEST_FILE, partition_dir
from privacy import GaussianActivationNoise, derive_seed
from split_learning.backbone import build_backbone
from split_learning.compute_peer import ComputePeer
from split_learning.config import OPTIMIZERS, optimizer_factory
from split_learning.institution import create_institutions
from split_learning.training import evaluate, train_batches
from tracking import TRACKING_KINDS, NullTracker, Tracker, open_tracker, source_tags

DEFAULT_OUTPUT = Path("experiments/results/noise_comparison.csv")


@dataclass(frozen=True)
class Settings:
    noise_stds: tuple[float, ...]
    rounds: int
    batches_per_turn: int
    batch_size: int
    optimizer: str
    learning_rate: float
    partition_seed: int
    noise_seed: int
    evaluation: str
    data_dir: str
    source_md5: str


@dataclass(frozen=True)
class TurnRow:
    noise_std: float
    evaluation: str
    round_number: int
    data_owner: str
    compute_peer: str
    backbone_version: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    activation_change_pct: float
    seconds: float


def run_condition(
    noise_std: float,
    settings: Settings,
    shards: dict[str, ImageLabelDataset],
    validation: ImageLabelDataset,
    log=print,
    tracker: Tracker | None = None,
) -> list[TurnRow]:
    tracker = tracker or NullTracker()
    with tracker.run(f"noise_std={noise_std:g}", {"noise_std": noise_std, "evaluation": settings.evaluation}, nested=True):
        rows = train_condition(noise_std, settings, shards, validation, log, tracker)
        final = [row for row in rows if row.round_number == settings.rounds]
        tracker.log_metrics(
            {
                "final_val_accuracy_mean": sum(row.validation_accuracy for row in final) / len(final),
                "mean_activation_change_pct": sum(row.activation_change_pct for row in rows) / len(rows),
            },
            step=settings.rounds,
        )
    return rows


def train_condition(
    noise_std: float,
    settings: Settings,
    shards: dict[str, ImageLabelDataset],
    validation: ImageLabelDataset,
    log,
    tracker: Tracker,
) -> list[TurnRow]:
    optimizer = optimizer_factory(settings.optimizer, settings.learning_rate)
    institution_ids = list(shards)
    institutions = create_institutions(institution_ids, MODEL_CONFIG, optimizer)
    peers = {institution_id: ComputePeer(MODEL_CONFIG, optimizer) for institution_id in institution_ids}
    canonical, version = build_backbone(MODEL_CONFIG).state_dict(), 0
    rows = []
    for turn in build_schedule(institution_ids, settings.rounds):
        started = time.monotonic()
        owner, peer = institutions[turn.data_owner_id], peers[turn.compute_peer_id]
        peer.load_backbone(canonical, str(version))
        batches = training_loader(
            shards[turn.data_owner_id], settings.batch_size, settings.partition_seed, turn.round_number, settings.batches_per_turn
        )
        noise = GaussianActivationNoise(noise_std, derive_seed(settings.noise_seed, turn.data_owner_id, turn.round_number))
        trained = train_batches(owner, peer, batches, perturb=noise)
        canonical, version = peer.backbone_update().state_dict, version + 1
        evaluation_noise = (
            GaussianActivationNoise(noise_std, derive_seed(settings.noise_seed, "evaluation", turn.data_owner_id, turn.round_number))
            if settings.evaluation == "noisy"
            else None
        )
        validated = evaluate(owner, peer, evaluation_loader(validation, 512), perturb=evaluation_noise)
        row = TurnRow(
            noise_std=noise_std,
            evaluation=settings.evaluation,
            round_number=turn.round_number,
            data_owner=turn.data_owner_id,
            compute_peer=turn.compute_peer_id,
            backbone_version=version,
            train_loss=trained.loss,
            train_accuracy=trained.accuracy,
            validation_loss=validated.loss,
            validation_accuracy=validated.accuracy,
            activation_change_pct=noise.summary().relative_change_pct,
            seconds=time.monotonic() - started,
        )
        log(
            f"noise {noise_std:g} round {row.round_number} {row.data_owner}->{row.compute_peer}: "
            f"train acc {row.train_accuracy:.3f}, val acc {row.validation_accuracy:.3f}, "
            f"activations changed {row.activation_change_pct:.1f}%"
        )
        tracker.log_metrics(
            {
                "train_loss": row.train_loss,
                "train_accuracy": row.train_accuracy,
                "val_loss": row.validation_loss,
                "val_accuracy": row.validation_accuracy,
                f"val_accuracy/{row.data_owner}": row.validation_accuracy,
                "activation_change_pct": row.activation_change_pct,
            },
            step=version,
        )
        rows.append(row)
    return rows


def inspect_activations(shards: dict[str, ImageLabelDataset], batch_size: int, seed: int) -> None:
    institution = create_institutions(["probe"], MODEL_CONFIG)["probe"]
    print(f"Head output shape per sample: {MODEL_CONFIG.head_output_shape} (untrained Head, seed {MODEL_CONFIG.seed})")
    stds = []
    for institution_id, shard in shards.items():
        images, _ = next(iter(training_loader(shard, batch_size, seed, 1, 1)))
        activations = institution.encode(images)
        stds.append(float(activations.std()))
        per_sample_norm = torch.linalg.vector_norm(activations.flatten(1), dim=1).mean()
        print(
            f"{institution_id}: mean {float(activations.mean()):.4f}  std {stds[-1]:.4f}  "
            f"max {float(activations.max()):.4f}  zero fraction {float((activations == 0).float().mean()):.2f}  "
            f"per-sample L2 {float(per_sample_norm):.3f}"
        )
    typical = sum(stds) / len(stds)
    print(f"typical activation std {typical:.4f}: noise_std of 0.5x and 2x that is {0.5 * typical:.3f} and {2 * typical:.3f}")


def write_results(rows: Sequence[TurnRow], settings: Settings, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    output.with_suffix(".settings.json").write_text(
        json.dumps({**asdict(settings), "model": repr(MODEL_CONFIG), "evaluation_split": "official validation"}, indent=2) + "\n"
    )


def print_summary(rows: Sequence[TurnRow], rounds: int) -> None:
    print("noise_std  final-round mean val acc  mean activation change")
    for noise_std in dict.fromkeys(row.noise_std for row in rows):
        final = [row for row in rows if row.noise_std == noise_std and row.round_number == rounds]
        condition = [row for row in rows if row.noise_std == noise_std]
        print(
            f"{noise_std:9g}  {sum(r.validation_accuracy for r in final) / len(final):24.3f}  "
            f"{sum(r.activation_change_pct for r in condition) / len(condition):21.1f}%"
        )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.noise_comparison")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--noise-std", type=float, nargs="+", default=[0.0])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batches-per-turn", type=int, default=351)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--evaluation", choices=["clean", "noisy"], default="clean")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tracking", choices=TRACKING_KINDS, default="mlflow")
    options = parser.parse_args(arguments)

    data_dir = options.data_dir or default_data_dir()
    shards = {
        institution_id: open_institution_training(data_dir, options.partition_seed, institution_id)
        for institution_id in DEFAULT_INSTITUTION_IDS
    }
    if options.inspect:
        inspect_activations(shards, options.batch_size, options.partition_seed)
        return 0

    manifest = PartitionManifest.read(partition_dir(data_dir, options.partition_seed) / MANIFEST_FILE)
    settings = Settings(
        noise_stds=tuple(options.noise_std),
        rounds=options.rounds,
        batches_per_turn=options.batches_per_turn,
        batch_size=options.batch_size,
        optimizer=options.optimizer,
        learning_rate=options.learning_rate,
        partition_seed=options.partition_seed,
        noise_seed=options.noise_seed,
        evaluation=options.evaluation,
        data_dir=str(data_dir),
        source_md5=manifest.source_md5,
    )
    validation = open_evaluation(data_dir, "val")
    tracker = open_tracker(options.tracking, "medweave-noise-comparison")
    params = {**asdict(settings), **{f"model.{key}": str(value) for key, value in asdict(MODEL_CONFIG).items()}}
    with tracker.run(f"noise-comparison-{settings.evaluation}-eval", params, tags=source_tags()):
        rows = [
            row
            for noise_std in settings.noise_stds
            for row in run_condition(noise_std, settings, shards, validation, tracker=tracker)
        ]
        write_results(rows, settings, options.output)
        for noise_std in settings.noise_stds:
            final = [row for row in rows if row.noise_std == noise_std and row.round_number == settings.rounds]
            tracker.log_metrics({f"final_val_accuracy/noise_{noise_std:g}": sum(r.validation_accuracy for r in final) / len(final)}, step=settings.rounds)
        for artifact in (options.output, options.output.with_suffix(".settings.json"), partition_dir(data_dir, options.partition_seed) / MANIFEST_FILE):
            tracker.log_artifact(artifact)
    print_summary(rows, settings.rounds)
    print(f"results: {options.output} and {options.output.with_suffix('.settings.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
