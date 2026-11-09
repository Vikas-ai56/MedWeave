"""One U-shaped training step with every party in the same process.

Used for single-laptop simulation and tests. In a distributed run the comments
mark where tensors cross the network via the coordinator relay.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer

from split_learning.backbone import Backbone
from split_learning.backward import backbone_backward, head_backward, tail_backward
from split_learning.forward import backbone_forward, head_forward, tail_forward
from split_learning.head import Head
from split_learning.tail import Tail

Perturb = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class Institution:
    head: Head
    tail: Tail
    head_optimizer: Optimizer
    tail_optimizer: Optimizer


@dataclass(frozen=True)
class ComputePeer:
    backbone: Backbone
    optimizer: Optimizer


def train_step(
    institution: Institution,
    peer: ComputePeer,
    inputs: Tensor,
    labels: Tensor,
    perturb: Perturb | None = None,
) -> float:
    smashed, head_pass = head_forward(institution.head, inputs)
    if perturb is not None:
        with torch.no_grad():
            smashed = perturb(smashed)

    # institution -> peer: smashed activations
    backbone_output, backbone_pass = backbone_forward(peer.backbone, smashed)

    # peer -> institution: backbone output
    tail_pass = tail_forward(institution.tail, backbone_output, labels)
    grad_backbone_output = tail_backward(tail_pass, institution.tail_optimizer)

    # institution -> peer: gradient w.r.t. backbone output
    grad_smashed = backbone_backward(backbone_pass, grad_backbone_output, peer.optimizer)

    # peer -> institution: gradient w.r.t. smashed activations
    head_backward(head_pass, grad_smashed, institution.head_optimizer)
    return tail_pass.loss.item()
