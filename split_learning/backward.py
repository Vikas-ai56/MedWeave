"""Backward half of the U-shaped split.

Each function back-propagates through one party's component, steps that
party's optimizer and, where needed, returns the gradient to send upstream.
Each component needs its own optimizer so stepping one never moves another.
"""

from torch import Tensor
from torch.optim import Optimizer

from split_learning.forward import BackbonePass, HeadPass, TailPass


def tail_backward(tail_pass: TailPass, optimizer: Optimizer) -> Tensor:
    """Return d(loss)/d(backbone output), to be sent to the compute peer."""
    optimizer.zero_grad()
    tail_pass.loss.backward()
    optimizer.step()
    return tail_pass.received.grad


def backbone_backward(
    backbone_pass: BackbonePass, grad_output: Tensor, optimizer: Optimizer
) -> Tensor:
    """Return d(loss)/d(smashed activations), to be sent back to the data owner."""
    optimizer.zero_grad()
    backbone_pass.output.backward(grad_output)
    optimizer.step()
    return backbone_pass.received.grad


def head_backward(head_pass: HeadPass, grad_activations: Tensor, optimizer: Optimizer) -> None:
    """Apply the returned gradient to the Head.

    The peer computed this gradient with respect to the *noisy* activations.
    It is applied unchanged to the clean Head output (straight-through), which
    is exact for additive noise since d(a + n)/da = I.
    """
    optimizer.zero_grad()
    head_pass.activations.backward(grad_activations)
    optimizer.step()
