"""Forward half of the U-shaped split.

Each function returns a detached tensor (what crosses the network) plus a pass
object holding the local autograd graph, which the matching backward function
in `split_learning.backward` consumes.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from split_learning.backbone import Backbone
from split_learning.head import Head
from split_learning.tail import Tail

LossFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class HeadPass:
    activations: Tensor


@dataclass(frozen=True)
class BackbonePass:
    received: Tensor
    output: Tensor


@dataclass(frozen=True)
class TailPass:
    received: Tensor
    logits: Tensor
    loss: Tensor


def head_forward(head: Head, inputs: Tensor) -> tuple[Tensor, HeadPass]:
    activations = head(inputs)
    return activations.detach(), HeadPass(activations)


def backbone_forward(backbone: Backbone, smashed: Tensor) -> tuple[Tensor, BackbonePass]:
    received = smashed.detach().requires_grad_()
    output = backbone(received)
    return output.detach(), BackbonePass(received, output)


def tail_forward(
    tail: Tail,
    backbone_output: Tensor,
    labels: Tensor,
    loss_fn: LossFn = F.cross_entropy,
) -> TailPass:
    received = backbone_output.detach().requires_grad_()
    logits = tail(received)
    return TailPass(received, logits, loss_fn(logits, labels))
