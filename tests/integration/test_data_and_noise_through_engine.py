"""Member 3's loaders and noise driven through Member 2's real engine with the coordinator's model config."""

import copy

import pytest
import torch

from coordinator.flower_app.protocol import MODEL_CONFIG
from institution_data import open_institution_training, training_loader
from privacy import GaussianActivationNoise
from split_learning.backbone import build_backbone
from split_learning.compute_peer import ComputePeer
from split_learning.institution import Institution
from split_learning.training import train_batches


class RecordingPeer(ComputePeer):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.received: list[torch.Tensor] = []

    def forward(self, batch_id, activations, backbone_version):
        self.received.append(activations.clone())
        return super().forward(batch_id, activations, backbone_version)


def run_turn(data_dir, noise_std: float):
    institution = Institution("inst-1", MODEL_CONFIG)
    peer = RecordingPeer(MODEL_CONFIG)
    peer.load_backbone(build_backbone(MODEL_CONFIG).state_dict(), "0")
    before = [copy.deepcopy(module.state_dict()) for module in (institution.head, peer.backbone, institution.tail)]
    batches = training_loader(open_institution_training(data_dir, 0, "inst-1"), 4, 0, round_number=1, num_batches=2)
    noise = GaussianActivationNoise(noise_std, seed=11)

    summary = train_batches(institution, peer, batches, perturb=noise)

    after = (institution.head, peer.backbone, institution.tail)
    changed = [any(not torch.equal(v, state[k]) for k, v in m.state_dict().items()) for m, state in zip(after, before)]
    return summary, peer, noise, changed


def test_model_config_matches_pathmnist():
    assert MODEL_CONFIG.input_shape == (3, 28, 28) and MODEL_CONFIG.num_classes == 9


@pytest.mark.parametrize("noise_std", [0.0, 0.5])
def test_full_split_step_updates_every_part_with_and_without_noise(provisioned_data_dir, noise_std):
    summary, peer, noise, changed = run_turn(provisioned_data_dir, noise_std)

    assert summary.num_batches == 2 and summary.num_samples == 8
    assert changed == [True, True, True]
    assert all(t.shape == (4, *MODEL_CONFIG.head_output_shape) for t in peer.received)
    assert len(noise.batch_stats) == 2


def test_peer_receives_the_noisy_activations(provisioned_data_dir):
    _, clean_peer, clean_noise, _ = run_turn(provisioned_data_dir, 0.0)
    _, noisy_peer, noisy_noise, _ = run_turn(provisioned_data_dir, 0.5)

    assert clean_noise.summary().relative_change_pct == 0.0
    assert noisy_noise.summary().relative_change_pct > 0.0
    assert not torch.equal(clean_peer.received[0], noisy_peer.received[0])
    assert all(t.is_floating_point() for t in noisy_peer.received)
