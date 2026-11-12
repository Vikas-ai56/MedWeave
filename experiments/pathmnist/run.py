"""MedWeavePathCNN on PathMNIST, split or centralized, logged to MLflow.

    python -m experiments.pathmnist.run --rounds 10 --seeds 0,1,2 --workers 3
    python -m experiments.pathmnist.run --rounds 10 --seeds 0,1,2 --workers 3 --centralized

Split mode: each round, every institution takes training turns on its own IID
partition while the next institution acts as its compute peer. The canonical
Backbone is handed from turn to turn as a checkpoint. A turn is a full local
epoch, or `--turn-steps` batches; either way one round is one pass over the
training data.

Centralized mode trains the same network unsplit on the whole training set
with the same recipe and sample budget; it is the ceiling for the architecture.

Both modes score validation after every round, keep the round with the best
mean validation accuracy and report the test split for that round only. Each
seed is a child MLflow run; the parent logs the mean (`{key}`) and sample
standard deviation (`std/{key}`) over seeds. See docs/accuracy_plan.md.
"""

import argparse
import copy
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from itertools import chain, islice, repeat
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.pathmnist.augment import Augment, dihedral_views
from experiments.pathmnist.data import (
    CLASS_NAMES,
    Augmentation,
    Split,
    batches,
    ensure_pathmnist,
    iid_partitions,
    load_splits,
)
from experiments.pathmnist.metrics import classification_metrics, confusion
from experiments.pathmnist.recipe import SCHEDULES, LRSchedule, StepClock, adamw
from split_learning.backbone import build_backbone
from split_learning.checkpoint import (
    load_institution_checkpoint,
    read_backbone_checkpoint,
    save_backbone_checkpoint,
    save_institution_checkpoint,
)
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.forward import LossFn
from split_learning.head import build_head
from split_learning.institution import Institution, StepMetrics, create_institutions
from split_learning.institution import metrics as step_metrics
from split_learning.path_cnn import path_cnn_v1_config, path_cnn_v2_config
from split_learning.tail import build_tail
from split_learning.training import Perturb, Summary, evaluation_mode, summarize, train_batches

INSTITUTION_IDS = ("inst-1", "inst-2", "inst-3", "inst-4")
CENTRAL_ID = "central"
EVAL_BATCH_SIZE = 256
REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
MODELS = {"v1": path_cnn_v1_config, "v2": path_cnn_v2_config}
NORMS = ("batch", "group")

Metrics = dict[str, float]
Forward = Callable[[Tensor], Tensor]


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part) for part in text.split(",") if part.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be a comma-separated list of distinct integers, got {text!r}")
    return seeds


@dataclass(frozen=True)
class RunConfig:
    rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    schedule: str = "constant"
    warmup: float = 0.05
    augment: bool = False
    jitter: float = 0.2
    tta: bool = False
    dropout: float = 0.2
    label_smoothing: float = 0.0
    model: str = "v1"
    norm: str = "batch"
    head_stages: int = 1
    turn_steps: int = 0
    head_lr_scale: float = 1.0
    carry_backbone_optimizer: bool = False
    noise_std: float = 0.0
    seeds: str = "0"
    workers: int = 1
    threads: int = 0
    centralized: bool = False
    phase: str = "0"
    label: str = ""
    data_dir: str = str(REPO_ROOT / "data")
    experiment: str = "medweave-pathmnist"

    def __post_init__(self) -> None:
        parse_seeds(self.seeds)
        choices = {"schedule": SCHEDULES, "model": tuple(MODELS), "norm": NORMS}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.rounds < 1 or self.local_epochs < 1 or self.batch_size < 1 or self.workers < 1:
            raise ValueError("rounds, local_epochs, batch_size and workers must be positive")
        if self.turn_steps < 0 or self.head_lr_scale < 0 or self.noise_std < 0 or self.threads < 0:
            raise ValueError("turn_steps, head_lr_scale, noise_std and threads cannot be negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")

    @property
    def mode(self) -> str:
        return "centralized" if self.centralized else "split"

    @property
    def run_name(self) -> str:
        name = f"{self.model}-{self.mode}-r{self.rounds}-seeds{self.seeds}"
        return f"{self.label}-{name}" if self.label else name

    @property
    def tags(self) -> dict[str, str]:
        return {"dataset": "PathMNIST", "mode": self.mode, "phase": self.phase, "label": self.label}


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in fields(RunConfig):
        flag = f"--{field.name.replace('_', '-')}"
        if isinstance(field.default, bool):
            parser.add_argument(flag, action="store_true")
        else:
            parser.add_argument(flag, type=type(field.default), default=field.default)
    return RunConfig(**vars(parser.parse_args(argv)))


# -------------------------------------------------------------------- recipe


def model_config(config: RunConfig, seed: int) -> SplitModelConfig:
    preset = MODELS[config.model](head_stages=config.head_stages, seed=seed)
    return replace(
        preset,
        tail_dropout=config.dropout,
        batch_norm=config.norm == "batch",
        group_norm=config.norm == "group",
    )


def loss_function(config: RunConfig) -> LossFn:
    return partial(F.cross_entropy, label_smoothing=config.label_smoothing)


def augmentation(config: RunConfig) -> Augmentation | None:
    return Augment(config.jitter) if config.augment else None


def gaussian_noise(std: float) -> Perturb | None:
    """Additive noise on outgoing activations, or None for no noise."""
    if std == 0:
        return None
    return lambda activations: activations + std * torch.randn_like(activations)


def epoch_stream(
    split: Split, epochs: int, batch_size: int, generator: torch.Generator, augment: Augmentation | None
) -> Iterator[tuple[Tensor, Tensor]]:
    return chain.from_iterable(batches(split, batch_size, generator, augment) for _ in range(epochs))


def combine(summaries: Sequence[Summary]) -> Summary:
    samples = sum(summary.num_samples for summary in summaries)
    return Summary(
        loss=sum(s.loss * s.num_samples for s in summaries) / samples,
        accuracy=sum(s.accuracy * s.num_samples for s in summaries) / samples,
        num_batches=sum(s.num_batches for s in summaries),
        num_samples=samples,
    )


# ---------------------------------------------------------------- evaluation


@contextmanager
def eval_mode(module: nn.Module) -> Iterator[None]:
    previous = [submodule.training for submodule in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for submodule, was_training in zip(module.modules(), previous):
            submodule.train(was_training)


def with_tta(forward: Forward) -> Forward:
    """Average the logits of the 8 orientations of each input."""
    return lambda inputs: torch.stack([forward(view) for view in dihedral_views(inputs)]).mean(dim=0)


def logits_for(forward: Forward, split: Split, batch_size: int = EVAL_BATCH_SIZE) -> Tensor:
    with torch.no_grad():
        return torch.cat([forward(inputs) for inputs, _ in batches(split, batch_size)])


def predict(
    institution: Institution,
    peer: ComputePeer,
    split: Split,
    batch_size: int = EVAL_BATCH_SIZE,
    tta: bool = False,
    perturb: Perturb | None = None,
) -> Tensor:
    """Logits for every sample in `split`, in order, with all modules in eval mode."""
    version = peer.version or ""

    def forward(inputs: Tensor) -> Tensor:
        activations = institution.encode(inputs)
        if perturb is not None:
            activations = perturb(activations)
        return institution.tail(peer.infer(activations, version))

    with evaluation_mode(institution, peer):
        return logits_for(with_tta(forward) if tta else forward, split, batch_size)


def predict_model(model: nn.Module, split: Split, tta: bool = False) -> Tensor:
    with eval_mode(model):
        return logits_for(with_tta(model) if tta else model, split)


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def scores(logits: Tensor, split: Split) -> Metrics:
    return classification_metrics(logits, split.labels, CLASS_NAMES)


def flatten(prefix: str, rows: Mapping[str, Metrics]) -> Metrics:
    """`{prefix}/{metric}/{id}` plus `{prefix}/{metric}/mean` across ids."""
    flat = {}
    for metric in next(iter(rows.values())):
        values = {id_: row[metric] for id_, row in rows.items()}
        flat |= {f"{prefix}/{metric}/{id_}": value for id_, value in values.items()}
        flat[f"{prefix}/{metric}/mean"] = statistics.fmean(values.values())
    return flat


def test_metrics(logits: Mapping[str, Tensor], split: Split, prefix: str = "test") -> Metrics:
    return flatten(prefix, {id_: scores(values, split) for id_, values in logits.items()})


def recalibrate_batchnorm(model: nn.Module, inputs: Iterable[Tensor]) -> None:
    """Replace every BatchNorm's running statistics with the average over `inputs`.

    Weights are not touched; modes and momenta are restored afterwards.
    """
    norms = [module for module in model.modules() if isinstance(module, BATCH_NORMS)]
    momenta = [norm.momentum for norm in norms]
    with eval_mode(model):
        for norm in norms:
            norm.reset_running_stats()
            norm.momentum = None  # cumulative average instead of an exponential one
            norm.train()
        try:
            with torch.no_grad():
                for batch in inputs:
                    model(batch)
        finally:
            for norm, momentum in zip(norms, momenta):
                norm.momentum = momentum


def split_diagnostics(
    institutions: Mapping[str, Institution],
    peer: ComputePeer,
    local_data: Mapping[str, Split],
    splits: Mapping[str, Split],
) -> Metrics:
    """Per institution, on the selected model.

    `diag/train_eval_accuracy`: eval-mode accuracy on its own training partition.
    `diag/recal_{val,test}_accuracy`: accuracy after recomputing the Head and
    Backbone BatchNorm statistics from that partition alone (on copies). A large
    gain means the shared running statistics, not the weights, cost accuracy.
    """
    rows = {}
    for id_, institution in institutions.items():
        data = local_data[id_]
        encoder = nn.Sequential(copy.deepcopy(institution.head), copy.deepcopy(peer.backbone))
        recalibrate_batchnorm(encoder, (inputs for inputs, _ in batches(data, EVAL_BATCH_SIZE)))
        recalibrated = nn.Sequential(encoder, institution.tail)
        rows[id_] = {
            "train_eval_accuracy": accuracy(predict(institution, peer, data), data.labels),
            "recal_val_accuracy": accuracy(predict_model(recalibrated, splits["val"]), splits["val"].labels),
            "recal_test_accuracy": accuracy(predict_model(recalibrated, splits["test"]), splits["test"].labels),
        }
    return flatten("diag", rows)


def model_params(config: RunConfig, model: SplitModelConfig, head: nn.Module, backbone: nn.Module, tail: nn.Module) -> dict[str, object]:
    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    return {
        "architecture": f"MedWeavePathCNN-{config.model}",
        "head_output_shape": model.head_output_shape,
        "backbone_output_shape": model.backbone_output_shape,
        "params_head": count(head),
        "params_backbone": count(backbone),
        "params_tail": count(tail),
        "optimizer": "AdamW",
    }


def log_test_report(path: Path, best_round: int, logits: Mapping[str, Tensor], split: Split) -> None:
    report = {
        "best_round": best_round,
        **{
            id_: {"metrics": scores(values, split), "confusion_matrix": confusion(values, split.labels)}
            for id_, values in logits.items()
        },
    }
    path.write_text(json.dumps(report, indent=2))
    mlflow.log_artifact(str(path))


class BestRound:
    """Tracks the round with the highest mean validation accuracy."""

    def __init__(self) -> None:
        self.round = 0
        self.validation: Metrics = {}

    def offer(self, round_: int, validation: Metrics) -> bool:
        key = "val/accuracy/mean"
        if self.validation and validation[key] <= self.validation[key]:
            return False
        self.round, self.validation = round_, validation
        return True

    def summary(self) -> Metrics:
        means = {f"best_{key}": value for key, value in self.validation.items() if key.endswith("/mean")}
        return means | {"best_round": float(self.round)}


# ----------------------------------------------------------------- split mode


def run_split(config: RunConfig, seed: int, splits: Mapping[str, Split], workdir: Path) -> Metrics:
    torch.manual_seed(seed)
    partitions = iid_partitions(len(splits["train"]), len(INSTITUTION_IDS), seed)
    local_data = {id_: splits["train"].subset(indices) for id_, indices in zip(INSTITUTION_IDS, partitions)}

    model = model_config(config, seed)
    optimizer = adamw(config.lr, config.weight_decay)
    institutions = create_institutions(INSTITUTION_IDS, model, optimizer, loss_function(config))
    if config.head_lr_scale != 1.0:
        head_optimizer = adamw(config.lr * config.head_lr_scale, config.weight_decay)
        for institution in institutions.values():
            institution.head_optimizer = head_optimizer(institution.head.parameters())
    peers = {id_: ComputePeer(model, optimizer) for id_ in INSTITUTION_IDS}
    evaluator = ComputePeer(model)
    canonical = build_backbone(model).state_dict()
    version = 0
    shuffle = torch.Generator().manual_seed(seed)
    augment, perturb = augmentation(config), gaussian_noise(config.noise_std)
    steps_per_round = config.local_epochs * sum(math.ceil(len(data) / config.batch_size) for data in local_data.values())
    clock = StepClock(LRSchedule(config.schedule, steps_per_round * config.rounds, config.warmup))
    backbone_optimizer_state = None

    first = institutions[INSTITUTION_IDS[0]]
    mlflow.log_params({"seed": seed} | model_params(config, model, first.head, evaluator.backbone, first.tail))
    mlflow.log_params({f"samples_{id_}": len(data) for id_, data in local_data.items()})
    mlflow.log_params({f"samples_{name}": len(split) for name, split in splits.items()})

    best = BestRound()
    started = time.perf_counter()
    for round_ in range(1, config.rounds + 1):
        streams = {
            id_: epoch_stream(local_data[id_], config.local_epochs, config.batch_size, shuffle, augment)
            for id_ in INSTITUTION_IDS
        }
        turns: list[Summary] = []
        active = list(INSTITUTION_IDS)
        while active:
            for owner_id in list(active):
                head_batch = next(streams[owner_id], None)
                if head_batch is None:
                    active.remove(owner_id)
                    continue
                rest = islice(streams[owner_id], config.turn_steps - 1) if config.turn_steps else streams[owner_id]
                owner = institutions[owner_id]
                peer = peers[INSTITUTION_IDS[(INSTITUTION_IDS.index(owner_id) + 1) % len(INSTITUTION_IDS)]]
                peer.load_backbone(canonical, str(version))
                if backbone_optimizer_state is not None:
                    peer.optimizer.load_state_dict(backbone_optimizer_state)

                scheduled = clock.drive(
                    chain([head_batch], rest), lambda: (owner.head_optimizer, owner.tail_optimizer, peer.optimizer)
                )
                summary = train_batches(owner, peer, scheduled, perturb)
                canonical, version = peer.backbone_update().state_dict, version + 1
                if config.carry_backbone_optimizer:
                    backbone_optimizer_state = copy.deepcopy(peer.optimizer.state_dict())
                turns.append(summary)
                mlflow.log_metrics(
                    {
                        f"train/loss/{owner_id}": summary.loss,
                        f"train/accuracy/{owner_id}": summary.accuracy,
                        "train/loss": summary.loss,
                        "train/accuracy": summary.accuracy,
                    },
                    step=version,
                )

        evaluator.load_backbone(canonical, str(version))
        validation = flatten(
            "val",
            {id_: scores(predict(inst, evaluator, splits["val"], perturb=perturb), splits["val"])
             for id_, inst in institutions.items()},
        )
        mlflow.log_metrics(validation, step=round_)
        if best.offer(round_, validation):
            save_backbone_checkpoint(workdir / "backbone_best.pt", canonical, version=str(version))
            for id_, institution in institutions.items():
                save_institution_checkpoint(workdir / f"{id_}_best.pt", institution)
        trained = combine(turns)
        print(
            f"seed {seed} round {round_} ({len(turns)} turns, backbone v{version}): "
            f"train loss {trained.loss:.4f} acc {trained.accuracy:.4f} | "
            f"val acc {validation['val/accuracy/mean']:.4f} auc {validation['val/auc/mean']:.4f} "
            f"(best: round {best.round}) [{time.perf_counter() - started:.0f}s]",
            flush=True,
        )

    train_seconds = time.perf_counter() - started
    evaluator.load_backbone(*read_backbone_checkpoint(workdir / "backbone_best.pt"))
    for id_, institution in institutions.items():
        load_institution_checkpoint(workdir / f"{id_}_best.pt", institution)

    test_logits = {id_: predict(inst, evaluator, splits["test"], perturb=perturb) for id_, inst in institutions.items()}
    final = (
        test_metrics(test_logits, splits["test"])
        | split_diagnostics(institutions, evaluator, local_data, splits)
        | best.summary()
        | {"train_seconds": train_seconds, "handoffs": float(version)}
    )
    log_test_report(workdir / "test_metrics.json", best.round, test_logits, splits["test"])
    if config.tta:
        tta_logits = {
            id_: predict(inst, evaluator, splits["test"], tta=True, perturb=perturb)
            for id_, inst in institutions.items()
        }
        final |= test_metrics(tta_logits, splits["test"], "test_tta")
        log_test_report(workdir / "test_tta_metrics.json", best.round, tta_logits, splits["test"])
    mlflow.log_metrics(final)
    mlflow.log_artifact(str(workdir / "backbone_best.pt"))
    return final


# ----------------------------------------------------------- centralized mode


def train_central_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    labels: Tensor,
    loss_fn: LossFn = F.cross_entropy,
) -> StepMetrics:
    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    return step_metrics(loss, logits, labels)


def run_centralized(config: RunConfig, seed: int, splits: Mapping[str, Split], workdir: Path) -> Metrics:
    torch.manual_seed(seed)
    model_cfg = model_config(config, seed)
    head, backbone, tail = build_head(model_cfg), build_backbone(model_cfg), build_tail(model_cfg)
    model = nn.Sequential(head, backbone, tail)
    optimizer = adamw(config.lr, config.weight_decay)(model.parameters())
    shuffle = torch.Generator().manual_seed(seed)
    augment, loss_fn = augmentation(config), loss_function(config)
    steps_per_round = config.local_epochs * math.ceil(len(splits["train"]) / config.batch_size)
    clock = StepClock(LRSchedule(config.schedule, steps_per_round * config.rounds, config.warmup))

    mlflow.log_params({"seed": seed} | model_params(config, model_cfg, head, backbone, tail))
    mlflow.log_params({f"samples_{name}": len(split) for name, split in splits.items()})

    best, best_state = BestRound(), copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    for round_ in range(1, config.rounds + 1):
        model.train()
        stream = epoch_stream(splits["train"], config.local_epochs, config.batch_size, shuffle, augment)
        steps = [
            train_central_step(model, optimizer, inputs, labels, loss_fn)
            for inputs, labels in clock.drive(stream, lambda: (optimizer,))
        ]
        summary = summarize(steps)
        # step = samples seen in units of one institution partition, matching split mode's Backbone version
        mlflow.log_metrics(
            {"train/loss": summary.loss, "train/accuracy": summary.accuracy},
            step=round_ * len(INSTITUTION_IDS),
        )
        validation = flatten("val", {CENTRAL_ID: scores(predict_model(model, splits["val"]), splits["val"])})
        mlflow.log_metrics(validation, step=round_)
        if best.offer(round_, validation):
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"seed {seed} round {round_} centralized: train loss {summary.loss:.4f} acc {summary.accuracy:.4f} | "
            f"val acc {validation['val/accuracy/mean']:.4f} (best: round {best.round}) "
            f"[{time.perf_counter() - started:.0f}s]",
            flush=True,
        )

    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    test_logits = {CENTRAL_ID: predict_model(model, splits["test"])}
    train_eval = accuracy(predict_model(model, splits["train"]), splits["train"].labels)
    final = (
        test_metrics(test_logits, splits["test"])
        | flatten("diag", {CENTRAL_ID: {"train_eval_accuracy": train_eval}})
        | best.summary()
        | {"train_seconds": train_seconds}
    )
    log_test_report(workdir / "test_metrics.json", best.round, test_logits, splits["test"])
    if config.tta:
        tta_logits = {CENTRAL_ID: predict_model(model, splits["test"], tta=True)}
        final |= test_metrics(tta_logits, splits["test"], "test_tta")
        log_test_report(workdir / "test_tta_metrics.json", best.round, tta_logits, splits["test"])
    mlflow.log_metrics(final)
    return final


# -------------------------------------------------------------------- driver


def aggregate(results: Sequence[Metrics]) -> Metrics:
    """Mean over seeds as `{key}`, sample standard deviation as `std/{key}`."""
    summary = {}
    for key in results[0]:
        values = [result[key] for result in results]
        summary[key] = statistics.fmean(values)
        summary[f"std/{key}"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def run_seed(config: RunConfig, seed: int, parent_run_id: str, tracking_uri: str) -> Metrics:
    """One seed as a child run of `parent_run_id`; safe to call in a worker process."""
    threads = config.threads or (max(1, (os.cpu_count() or 1) // config.workers) if config.workers > 1 else 0)
    if threads:
        torch.set_num_threads(threads)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment)
    splits = load_splits(ensure_pathmnist(Path(config.data_dir)))
    runner = run_centralized if config.centralized else run_split
    with (
        mlflow.start_run(
            run_name=f"seed-{seed}",
            nested=True,
            parent_run_id=parent_run_id,
            tags=config.tags | {"seed": str(seed)},
        ),
        tempfile.TemporaryDirectory() as tmp,
    ):
        return runner(config, seed, splits, Path(tmp))


HEADLINE = (
    "best_val/accuracy/mean",
    "test/accuracy/mean",
    "test/balanced_accuracy/mean",
    "test/macro_f1/mean",
    "test/auc/mean",
    "test_tta/accuracy/mean",
    "test_tta/auc/mean",
    "diag/train_eval_accuracy/mean",
    "diag/recal_val_accuracy/mean",
    "diag/recal_test_accuracy/mean",
    "best_round",
)


def main() -> None:
    config = parse_args()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment)
    ensure_pathmnist(Path(config.data_dir))
    seeds = parse_seeds(config.seeds)

    with mlflow.start_run(run_name=config.run_name, tags=config.tags) as parent:
        mlflow.log_params(asdict(config))
        run = partial(run_seed, parent_run_id=parent.info.run_id, tracking_uri=tracking_uri)
        if config.workers > 1:
            with ProcessPoolExecutor(min(config.workers, len(seeds))) as pool:
                results = list(pool.map(run, repeat(config), seeds))
        else:
            results = [run(config, seed) for seed in seeds]
        summary = aggregate(results)
        mlflow.log_metrics(summary)

    print(f"\n{config.run_name}: mean ± std over seeds {config.seeds}")
    for key in HEADLINE:
        if key in summary:
            print(f"  {key:32s} {summary[key]:.4f} ± {summary[f'std/{key}']:.4f}")


if __name__ == "__main__":
    main()
