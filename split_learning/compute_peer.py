from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from split_learning.backbone import build_backbone
from split_learning.backward import backbone_backward
from split_learning.checkpoint import StateDict, clone_state, load_module_state
from split_learning.config import OptimizerFactory, SplitModelConfig, default_optimizer
from split_learning.errors import ProtocolError
from split_learning.forward import BackbonePass, backbone_forward
from split_learning.validation import PendingBatch, check_batch, check_gradient


@dataclass(frozen=True)
class BackboneUpdate:
    """Backbone state after a training turn, returned to the coordinator."""

    base_version: str
    state_dict: StateDict
    num_batches: int


class ComputePeer:
    """Runs the shared Backbone for one data-owning institution's training turn.

    Lifecycle: `load_backbone`, then `forward` and `backward` once per batch,
    then `backbone_update`. The peer only ever receives activations and
    gradients; inputs and labels never reach it. The methods take and return
    plain tensors and strings so a service can call them across processes.
    """

    def __init__(
        self, config: SplitModelConfig, optimizer_factory: OptimizerFactory = default_optimizer
    ) -> None:
        self.config = config
        self.backbone = build_backbone(config)
        self._optimizer_factory = optimizer_factory
        self.optimizer = optimizer_factory(self.backbone.parameters())
        self.version: str | None = None
        self._pending: PendingBatch[BackbonePass] = PendingBatch()
        self._num_batches = 0

    def load_backbone(self, state_dict: Mapping[str, Tensor], version: str) -> None:
        """Start a turn from a coordinator checkpoint. Optimizer state starts fresh."""
        load_module_state(self.backbone, state_dict, "Backbone")
        self.optimizer = self._optimizer_factory(self.backbone.parameters())
        self.version = version
        self._pending.clear()
        self._num_batches = 0

    def forward(self, batch_id: str, activations: Tensor, backbone_version: str) -> Tensor:
        """Run the Backbone on received activations and keep the graph for `backward`."""
        self._check_version(backbone_version)
        check_batch(activations, self.config.head_output_shape, "activations")
        output, backbone_pass = backbone_forward(self.backbone, activations)
        self._pending.put(batch_id, backbone_pass)
        return output

    def backward(self, batch_id: str, gradient: Tensor, backbone_version: str) -> Tensor:
        """Update the Backbone and return the gradient for the institution's Head."""
        self._check_version(backbone_version)
        backbone_pass = self._pending.peek(batch_id)
        check_gradient(gradient, backbone_pass.output.shape, "gradient")
        self._pending.take(batch_id)
        input_gradient = backbone_backward(backbone_pass, gradient, self.optimizer)
        self._num_batches += 1
        return input_gradient

    def infer(self, activations: Tensor, backbone_version: str) -> Tensor:
        """Forward pass for evaluation: no graph is kept and nothing is updated."""
        self._check_version(backbone_version)
        check_batch(activations, self.config.head_output_shape, "activations")
        with torch.no_grad():
            return self.backbone(activations)

    def backbone_update(self) -> BackboneUpdate:
        self._check_loaded()
        if self._pending.batch_id is not None:
            raise ProtocolError(f"batch {self._pending.batch_id!r} has not finished its backward pass")
        return BackboneUpdate(self.version, clone_state(self.backbone), self._num_batches)

    def discard_pending(self) -> None:
        """Drop an unfinished batch, e.g. after the data owner disconnects."""
        self._pending.clear()

    def train(self, mode: bool = True) -> "ComputePeer":
        self.backbone.train(mode)
        return self

    def eval(self) -> "ComputePeer":
        return self.train(False)

    def _check_loaded(self) -> None:
        if self.version is None:
            raise ProtocolError("no Backbone checkpoint loaded; call load_backbone first")

    def _check_version(self, backbone_version: str) -> None:
        self._check_loaded()
        if backbone_version != self.version:
            raise ProtocolError(
                f"payload is for Backbone version {backbone_version!r}, peer has {self.version!r}"
            )
