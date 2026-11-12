import threading

import pytest
import torch

from coordinator.flower_app.roles import run_owner_turn, run_peer_turn
from coordinator.relay.relay import Kind, Relay, RelayTimeout, decode_tensor
from split_learning.backbone import build_backbone
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.errors import ProtocolError
from split_learning.institution import create_institutions
from split_learning.training import train_batches

OWNER, PEER, VERSION = "inst-1", "inst-2", "0"


@pytest.fixture
def config() -> SplitModelConfig:
    return SplitModelConfig(
        in_channels=1, image_size=(8, 8), head_channels=(4,), backbone_channels=(6,), num_classes=3
    )


@pytest.fixture
def batches(config):
    generator = torch.Generator().manual_seed(0)
    return [
        (torch.randn(4, 1, 8, 8, generator=generator), torch.randint(0, 3, (4,), generator=generator))
        for _ in range(3)
    ]


class RecordingRelay(Relay):
    """Relay that remembers what crossed it, to check that labels never do."""

    def __init__(self) -> None:
        super().__init__()
        self.sent = []

    def send(self, recipient, envelope):
        self.sent.append((recipient, envelope))
        super().send(recipient, envelope)


def make_parties(config):
    institution = create_institutions([OWNER], config)[OWNER]
    peer = ComputePeer(config)
    peer.load_backbone(build_backbone(config).state_dict(), VERSION)
    return institution, peer


def run_turn(institution, peer, batches, relay, timeout=5.0):
    """Run both roles concurrently, like two machines would."""
    outcome = {}

    def peer_side():
        try:
            outcome["update"] = run_peer_turn(peer, relay, PEER, timeout)
        except Exception as error:
            outcome["peer_error"] = error

    thread = threading.Thread(target=peer_side)
    thread.start()
    try:
        outcome["summary"] = run_owner_turn(institution, batches, relay, PEER, VERSION, timeout)
    except Exception as error:
        outcome["owner_error"] = error
    thread.join(timeout=10)
    assert not thread.is_alive(), "compute peer did not stop"
    return outcome


@pytest.fixture
def relay():
    relay = RecordingRelay()
    relay.open_turn(OWNER, PEER, VERSION)
    return relay


def test_relayed_turn_matches_in_process_training(config, batches, relay):
    institution, peer = make_parties(config)
    outcome = run_turn(institution, peer, batches, relay)

    twin_institution, twin_peer = make_parties(config)
    expected = train_batches(twin_institution, twin_peer, batches)

    assert outcome["summary"].loss == pytest.approx(expected.loss)
    assert outcome["summary"].accuracy == expected.accuracy
    update, twin_update = outcome["update"], twin_peer.backbone_update()
    assert update.num_batches == len(batches)
    assert update.base_version == VERSION
    for key, tensor in twin_update.state_dict.items():
        assert torch.allclose(update.state_dict[key], tensor), key
    for key, tensor in twin_institution.head.state_dict().items():
        assert torch.allclose(institution.head.state_dict()[key], tensor), key


def test_only_activations_and_gradients_cross_the_relay(config, batches, relay):
    institution, peer = make_parties(config)
    run_turn(institution, peer, batches, relay)

    kinds = [envelope.kind for _, envelope in relay.sent]
    per_batch = [Kind.ACTIVATIONS, Kind.BACKBONE_OUTPUT, Kind.GRADIENT, Kind.INPUT_GRADIENT]
    assert kinds == per_batch * len(batches) + [Kind.DONE]
    for _, envelope in relay.sent:
        assert envelope.institution_id == OWNER
        if envelope.kind is not Kind.DONE:
            assert decode_tensor(envelope.body).is_floating_point()  # labels are integer tensors


def test_perturb_is_applied_to_activations_before_sending(config, batches, relay):
    institution, peer = make_parties(config)
    perturb_calls = []

    def perturb(activations):
        perturb_calls.append(activations.shape)
        return activations + 1000.0

    thread = threading.Thread(target=run_peer_turn, args=(peer, relay, PEER, 5.0))
    thread.start()
    run_owner_turn(institution, batches, relay, PEER, VERSION, 5.0, perturb=perturb)
    thread.join(timeout=10)

    assert len(perturb_calls) == len(batches)
    sent_activations = [e for _, e in relay.sent if e.kind is Kind.ACTIVATIONS]
    assert all(decode_tensor(e.body).min() > 500 for e in sent_activations)


def test_owner_failure_still_stops_the_peer(config, batches, relay):
    institution, peer = make_parties(config)
    inputs, labels = batches[0]
    bad_batches = [(inputs, labels.float())]  # wrong label dtype fails at the Tail

    outcome = run_turn(institution, peer, bad_batches, relay)

    assert "owner_error" in outcome
    assert isinstance(outcome["peer_error"], ProtocolError)  # unfinished batch, no update returned
    assert relay.sent[-1][1].kind is Kind.DONE


def test_peer_gives_up_when_owner_never_sends(config, relay):
    _, peer = make_parties(config)
    with pytest.raises(RelayTimeout):
        run_peer_turn(peer, relay, PEER, timeout=0.1)
