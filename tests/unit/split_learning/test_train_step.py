import copy
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from split_learning.backbone import build_backbone
from split_learning.checkpoint import get_weights, weights_to_state
from split_learning.config import SplitModelConfig
from split_learning.head import build_head
from split_learning.step import ComputePeer, Institution, train_step
from split_learning.tail import build_tail

LEARNING_RATE = 0.1


@pytest.fixture
def make_institution(config: SplitModelConfig) -> Callable[[], Institution]:
    def make() -> Institution:
        head, tail = build_head(config), build_tail(config)
        return Institution(
            head=head,
            tail=tail,
            head_optimizer=torch.optim.SGD(head.parameters(), lr=LEARNING_RATE),
            tail_optimizer=torch.optim.SGD(tail.parameters(), lr=LEARNING_RATE),
        )

    return make


@pytest.fixture
def make_peer(config: SplitModelConfig) -> Callable[[], ComputePeer]:
    def make() -> ComputePeer:
        backbone = build_backbone(config)
        return ComputePeer(backbone, torch.optim.SGD(backbone.parameters(), lr=LEARNING_RATE))

    return make


def snapshot(module: nn.Module) -> dict[str, Tensor]:
    return copy.deepcopy(module.state_dict())


def differs(module: nn.Module, state: dict[str, Tensor]) -> bool:
    return any(not torch.equal(p, state[k]) for k, p in module.state_dict().items())


def test_train_step_updates_head_backbone_and_tail(make_institution, make_peer, batch) -> None:
    institution, peer = make_institution(), make_peer()
    before = [snapshot(m) for m in (institution.head, peer.backbone, institution.tail)]

    loss = train_step(institution, peer, *batch)

    assert loss > 0
    for module, state in zip((institution.head, peer.backbone, institution.tail), before):
        assert differs(module, state), type(module).__name__


def test_repeated_steps_reduce_loss(make_institution, make_peer, batch) -> None:
    institution, peer = make_institution(), make_peer()
    losses = [train_step(institution, peer, *batch) for _ in range(30)]
    assert losses[-1] < losses[0]


def test_perturb_is_applied_to_smashed_activations(make_institution, make_peer, batch) -> None:
    seen: list[Tensor] = []

    def record(smashed: Tensor) -> Tensor:
        seen.append(smashed)
        return smashed + 0.01 * torch.randn_like(smashed)

    train_step(make_institution(), make_peer(), *batch, perturb=record)

    assert len(seen) == 1 and not seen[0].requires_grad


def test_sequential_turns_share_backbone_but_keep_local_models(
    make_institution, make_peer, batch
) -> None:
    first, second = make_institution(), make_institution()
    inputs, labels = batch
    second_inputs = torch.flip(inputs, dims=[0])

    peer_a = make_peer()
    train_step(first, peer_a, inputs, labels)
    checkpoint = get_weights(peer_a.backbone)

    peer_b = make_peer()
    peer_b.backbone.load_state_dict(weights_to_state(peer_b.backbone, checkpoint))
    backbone_received = snapshot(peer_b.backbone)
    second_head_before = snapshot(second.head)
    first_head_after_turn = snapshot(first.head)
    train_step(second, peer_b, second_inputs, labels)

    assert not differs(peer_a.backbone, backbone_received)
    assert differs(peer_b.backbone, backbone_received)
    assert differs(second.head, second_head_before)
    assert not differs(first.head, first_head_after_turn)
    assert differs(first.head, snapshot(second.head))
