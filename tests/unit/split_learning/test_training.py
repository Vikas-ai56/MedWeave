import copy
import gc
import inspect
import weakref

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from split_learning.backbone import build_backbone
from split_learning.checkpoint import StateDict
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.errors import ProtocolError, ShapeError
from split_learning.institution import Institution, StepMetrics, create_institutions
from split_learning.training import evaluate, summarize, train_batches, train_step

IDS = ["inst-1", "inst-2", "inst-3", "inst-4"]


def sgd(parameters) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=0.1)


@pytest.fixture
def institution(config: SplitModelConfig) -> Institution:
    return Institution("inst-1", config, sgd)


@pytest.fixture
def peer(config: SplitModelConfig) -> ComputePeer:
    peer = ComputePeer(config, sgd)
    peer.load_backbone(build_backbone(config).state_dict(), "0")
    return peer


def synthetic_batches(config: SplitModelConfig, count: int, seed: int = 0) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(4, *config.input_shape, generator=generator),
            torch.randint(0, config.num_classes, (4,), generator=generator),
        )
        for _ in range(count)
    ]


def snapshot(*modules: nn.Module) -> list[StateDict]:
    return [copy.deepcopy(module.state_dict()) for module in modules]


def changed(module: nn.Module, state: StateDict) -> bool:
    return any(not torch.equal(value, state[key]) for key, value in module.state_dict().items())


def test_train_step_updates_head_backbone_and_tail(institution, peer, batch) -> None:
    modules = (institution.head, peer.backbone, institution.tail)
    before = snapshot(*modules)

    metrics = train_step(institution, peer, *batch)

    assert metrics.total == 4 and metrics.loss > 0
    for module, state in zip(modules, before):
        assert changed(module, state), type(module).__name__


def test_consecutive_batches_use_fresh_gradients(institution, peer, config) -> None:
    first, second = synthetic_batches(config, 2)
    train_step(institution, peer, *first, batch_id="b0")
    reference = nn.Sequential(
        copy.deepcopy(institution.head), copy.deepcopy(peer.backbone), copy.deepcopy(institution.tail)
    )

    train_step(institution, peer, *second, batch_id="b1")
    F.cross_entropy(reference(second[0]), second[1]).backward()

    split_params = [*institution.head.parameters(), *peer.backbone.parameters(), *institution.tail.parameters()]
    for split, ref in zip(split_params, reference.parameters()):
        assert torch.allclose(split.grad, ref.grad, atol=1e-6)


def test_multiple_batches_reduce_loss(institution, peer, config) -> None:
    batches = synthetic_batches(config, 4)
    first = train_batches(institution, peer, batches)
    for _ in range(15):
        last = train_batches(institution, peer, batches)

    assert first.num_batches == 4 and first.num_samples == 16
    assert last.loss < first.loss
    assert peer.backbone_update().num_batches == 64


def test_no_batch_is_left_pending_after_training(institution, peer, config) -> None:
    train_batches(institution, peer, synthetic_batches(config, 3))

    assert peer.backbone_update().num_batches == 3
    with pytest.raises(ProtocolError, match="no forward pass"):
        institution.head_backward("inst-1-2", torch.randn(4, *config.head_output_shape))


def test_failed_step_is_discarded_by_both_parties(institution, peer, batch) -> None:
    inputs, labels = batch
    backbone_before = snapshot(peer.backbone)[0]

    with pytest.raises(ShapeError, match="labels"):
        train_step(institution, peer, inputs, labels.float())

    assert not changed(peer.backbone, backbone_before)
    train_step(institution, peer, inputs, labels)


def test_perturb_is_applied_to_outgoing_activations(institution, peer, batch) -> None:
    seen: list[Tensor] = []

    def record(activations: Tensor) -> Tensor:
        seen.append(activations)
        return activations + 0.01 * torch.randn_like(activations)

    train_step(institution, peer, *batch, perturb=record)

    assert len(seen) == 1 and not seen[0].requires_grad


def test_labels_never_reach_the_compute_peer(institution, peer, batch) -> None:
    inputs, labels = batch
    received: list[object] = []

    class SpyPeer(ComputePeer):
        def forward(self, *args):
            received.extend(args)
            return super().forward(*args)

        def backward(self, *args):
            received.extend(args)
            return super().backward(*args)

    spy = SpyPeer(peer.config, sgd)
    spy.load_backbone(peer.backbone.state_dict(), "0")
    train_step(institution, spy, inputs, labels)

    tensors = [arg for arg in received if isinstance(arg, Tensor)]
    assert len(tensors) == 2
    assert all(t.is_floating_point() and t.ndim == 4 for t in tensors)
    assert not any(t is labels or t is inputs for t in tensors)
    for method in (ComputePeer.forward, ComputePeer.backward, ComputePeer.infer):
        assert "labels" not in inspect.signature(method).parameters


def test_evaluate_updates_nothing_and_restores_modes(institution, peer, config) -> None:
    modules = (institution.head, peer.backbone, institution.tail)
    before = snapshot(*modules)
    modes_seen: list[bool] = []
    hook = institution.tail.register_forward_hook(lambda module, *_: modes_seen.append(module.training))

    summary = evaluate(institution, peer, synthetic_batches(config, 2))
    hook.remove()

    assert summary.num_samples == 8 and 0.0 <= summary.accuracy <= 1.0
    assert modes_seen == [False, False]
    assert all(module.training for module in modules)
    for module, state in zip(modules, before):
        assert not changed(module, state)
    assert peer.backbone_update().num_batches == 0


def test_evaluate_restores_eval_mode_if_already_set(institution, peer, config) -> None:
    institution.eval()
    evaluate(institution, peer, synthetic_batches(config, 1))
    assert not institution.head.training and peer.backbone.training


def test_summarize_weights_by_batch_size() -> None:
    summary = summarize([StepMetrics(loss=1.0, correct=1, total=1), StepMetrics(loss=4.0, correct=3, total=3)])

    assert summary.loss == pytest.approx(13 / 4)
    assert summary.accuracy == 1.0
    assert summary.num_batches == 2 and summary.num_samples == 4


def test_summarize_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize([])


def test_four_institutions_take_sequential_turns_on_one_backbone(config: SplitModelConfig) -> None:
    institutions = create_institutions(IDS, config, sgd)
    peers = {id_: ComputePeer(config, sgd) for id_ in IDS}
    canonical, version = build_backbone(config).state_dict(), 0
    initial_backbone = copy.deepcopy(canonical)

    for turn, owner_id in enumerate(IDS):
        compute_peer = peers[IDS[(turn + 1) % len(IDS)]]
        owner = institutions[owner_id]
        others_before = {id_: snapshot(i.head, i.tail) for id_, i in institutions.items() if id_ != owner_id}

        compute_peer.load_backbone(canonical, str(version))
        train_batches(owner, compute_peer, synthetic_batches(config, 2, seed=turn))
        update = compute_peer.backbone_update()

        assert update.base_version == str(version) and update.num_batches == 2
        assert any(not torch.equal(update.state_dict[k], canonical[k]) for k in canonical)
        for id_, (head_state, tail_state) in others_before.items():
            assert not changed(institutions[id_].head, head_state), id_
            assert not changed(institutions[id_].tail, tail_state), id_
        canonical, version = update.state_dict, version + 1

    assert version == 4
    assert any(not torch.equal(canonical[k], initial_backbone[k]) for k in canonical)
    heads = [institutions[id_].head.state_dict() for id_ in IDS]
    for other in heads[1:]:
        assert any(not torch.equal(heads[0][k], other[k]) for k in other)


def test_step_with_unloaded_peer_fails_cleanly(institution, config, batch) -> None:
    with pytest.raises(ProtocolError, match="load_backbone"):
        train_step(institution, ComputePeer(config), *batch)
    institution.head_forward("next", batch[0])


def test_computation_graphs_are_freed_after_each_batch(institution, peer, config) -> None:
    outputs: list[weakref.ref] = []

    def track(module, args, output) -> None:
        outputs.append(weakref.ref(output))

    hooks = [institution.head.register_forward_hook(track), peer.backbone.register_forward_hook(track)]
    train_batches(institution, peer, synthetic_batches(config, 3))
    for hook in hooks:
        hook.remove()
    gc.collect()

    assert len(outputs) == 6
    assert all(ref() is None for ref in outputs)
