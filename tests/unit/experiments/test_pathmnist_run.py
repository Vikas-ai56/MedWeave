"""Phase 0 measurement harness: seeds, best-round selection, BatchNorm diagnostics."""

import mlflow
import pytest
import torch
from torch import nn

from experiments.pathmnist.data import Split
from experiments.pathmnist.run import (
    BestRound,
    RunConfig,
    aggregate,
    parse_args,
    parse_seeds,
    predict_model,
    recalibrate_batchnorm,
    run_centralized,
    run_split,
    train_central_step,
)
from split_learning.path_cnn import path_cnn_v1_config
from split_learning.backbone import build_backbone
from split_learning.head import build_head
from split_learning.tail import build_tail


def synthetic_split(size: int, seed: int) -> Split:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (size, 3, 28, 28), dtype=torch.uint8, generator=generator)
    return Split(images, torch.arange(size) % 9)


def test_parse_seeds() -> None:
    assert parse_seeds("0,1, 2") == (0, 1, 2)
    for bad in ("", "1,1", "a"):
        with pytest.raises(ValueError):
            parse_seeds(bad)


def test_parse_args_reads_flags_and_validates_seeds() -> None:
    config = parse_args(["--rounds", "5", "--seeds", "0,1", "--centralized", "--phase", "1"])

    assert (config.rounds, config.seeds, config.centralized, config.phase) == (5, "0,1", True, "1")
    assert parse_args([]).centralized is False and parse_args([]).mode == "split"
    with pytest.raises(ValueError, match="seeds"):
        RunConfig(seeds="")


def test_best_round_keeps_the_highest_mean_validation_accuracy() -> None:
    best = BestRound()

    assert best.offer(1, {"val/accuracy/mean": 0.8, "val/accuracy/a": 0.8})
    assert best.offer(2, {"val/accuracy/mean": 0.9, "val/accuracy/a": 0.9})
    assert not best.offer(3, {"val/accuracy/mean": 0.9, "val/accuracy/a": 0.7})
    assert best.summary() == {"best_val/accuracy/mean": 0.9, "best_round": 2.0}


def test_aggregate_reports_mean_and_sample_std() -> None:
    summary = aggregate([{"acc": 0.7}, {"acc": 0.8}, {"acc": 0.9}])

    assert summary["acc"] == pytest.approx(0.8)
    assert summary["std/acc"] == pytest.approx(0.1)
    assert aggregate([{"acc": 0.5}]) == {"acc": 0.5, "std/acc": 0.0}


def test_recalibration_sets_running_stats_to_the_data_average() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Dropout(0.5))
    model.train()
    inputs = [torch.randn(8, 3, 5, 5) + 3.0 for _ in range(2)]
    weights_before = [p.clone() for p in model.parameters()]

    recalibrate_batchnorm(model, inputs)

    with torch.no_grad():
        features = model[0](torch.cat(inputs))
    norm = model[1]
    assert torch.allclose(norm.running_mean, features.mean(dim=(0, 2, 3)), atol=1e-5)
    assert norm.momentum == 0.1 and all(module.training for module in model.modules())
    assert all(torch.equal(a, b) for a, b in zip(weights_before, model.parameters()))


def test_central_step_updates_the_model_and_predict_restores_modes() -> None:
    config = path_cnn_v1_config()
    model = nn.Sequential(build_head(config), build_backbone(config), build_tail(config))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.clone() for p in model.parameters()]
    inputs, labels = torch.randn(8, 3, 28, 28), torch.arange(8) % 9

    step = train_central_step(model, optimizer, inputs, labels)
    logits = predict_model(model, synthetic_split(10, seed=0))

    assert step.total == 8 and step.loss > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert logits.shape == (10, 9) and model.training


FULL_RECIPE = {
    "augment": True,
    "weight_decay": 1e-4,
    "schedule": "cosine",
    "tta": True,
    "dropout": 0.3,
    "label_smoothing": 0.1,
}
SPLIT_ONLY = {"turn_steps": 1, "head_lr_scale": 0.1, "carry_backbone_optimizer": True, "noise_std": 0.1}


def tiny_run(runner, tmp_path, monkeypatch, **options):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    splits = {"train": synthetic_split(32, 0), "val": synthetic_split(18, 1), "test": synthetic_split(18, 2)}
    workdir = tmp_path / "work"
    workdir.mkdir()
    with mlflow.start_run():
        final = runner(RunConfig(**({"rounds": 2, "batch_size": 8} | options)), 0, splits, workdir)
        artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(mlflow.active_run().info.run_id)}
    return final, artifacts


@pytest.mark.parametrize(
    "runner, options",
    [
        (run_split, {}),
        (run_centralized, {}),
        (run_split, FULL_RECIPE | SPLIT_ONLY | {"model": "v2", "norm": "group"}),
        (run_centralized, FULL_RECIPE | {"model": "v2"}),
    ],
    ids=["split", "centralized", "split-everything", "centralized-everything"],
)
def test_runner_end_to_end_on_tiny_synthetic_data(runner, options, tmp_path, monkeypatch) -> None:
    final, artifacts = tiny_run(runner, tmp_path, monkeypatch, **options)

    assert final["best_round"] in (1.0, 2.0)
    assert 0.0 <= final["test/accuracy/mean"] <= 1.0
    assert "test/recall_STR/mean" in final and "diag/train_eval_accuracy/mean" in final
    assert "test_metrics.json" in artifacts
    assert ("test_tta/accuracy/mean" in final) == options.get("tta", False)
    if runner is run_split:
        assert {"diag/recal_val_accuracy/inst-4", "diag/recal_test_accuracy/mean"} <= set(final)
        assert "backbone_best.pt" in artifacts
        # 8 samples per institution at batch 8: one batch each, so one handoff per institution per round
        assert final["handoffs"] == 8.0


def test_turn_steps_hands_over_after_every_n_batches(tmp_path, monkeypatch) -> None:
    final, _ = tiny_run(run_split, tmp_path, monkeypatch, batch_size=2, turn_steps=2)

    # 8 samples per institution at batch 2 = 4 batches = 2 turns, x 4 institutions x 2 rounds
    assert final["handoffs"] == 16.0


def test_head_lr_scale_zero_freezes_every_head(tmp_path, monkeypatch) -> None:
    captured = {}
    original = run_split.__globals__["create_institutions"]

    def spy(*args, **kwargs):
        captured.update(original(*args, **kwargs))
        return captured

    monkeypatch.setitem(run_split.__globals__, "create_institutions", spy)
    before = path_cnn_v1_config()
    tiny_run(run_split, tmp_path, monkeypatch, head_lr_scale=0.0)

    initial = build_head(before).state_dict()
    for institution in captured.values():
        for name, weight in institution.head.named_parameters():
            assert torch.equal(weight, initial[name]), name
