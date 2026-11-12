# MedWeave — Shared Interfaces

This document defines the initial contracts between the main MedWeave components.

Implementations may change internally, but public interfaces must remain compatible.

## 1. Main Components

| Component | Responsibility |
|---|---|
| Coordinator | Training schedule, institution management and Backbone checkpoint coordination |
| Institutional Client | Local data, Head, Tail, labels and local optimization |
| Compute Peer | Remote Backbone computation |
| Privacy Layer | Activation noise before transmission |
| Dashboard | Monitoring and visualization |

## 2. Model Architecture

MedWeave uses U-shaped Split Learning.

Each institution owns:
- A local Head.
- A local Tail.
- Its own dataset and labels.
- Its own Head and Tail optimizer states.

The shared Backbone is maintained as one canonical model, with its latest checkpoint managed by the coordinator.

A compute peer temporarily loads the current Backbone checkpoint to perform the assigned computation.

## 3. Model Interfaces

All model components should be implemented as standard PyTorch `nn.Module` classes.

Expected components:

```python
class Head(nn.Module):
    def forward(self, x):
        ...

class Backbone(nn.Module):
    def forward(self, activations):
        ...

class Tail(nn.Module):
    def forward(self, activations):
        ...
```

The architecture must support configurable:
- Input channels and dimensions.
- Number of output classes.
- Hidden dimensions or channel widths.
- Model split configuration.

The final dataset and architecture-specific dimensions will be finalized separately.

## 4. U-Shaped Forward Pass

The training flow is:

1. The institution sends its local batch through the Head.
2. The Head produces intermediate activations.
3. The Privacy Layer optionally adds configured noise.
4. The activations are sent to the assigned compute peer.
5. The compute peer runs the Backbone.
6. Backbone output activations are returned to the institution.
7. The institution runs the Tail.
8. The institution calculates the loss using its local labels.

Labels must never be sent to the compute peer.

## 5. Backward Pass

The backward flow is:

1. The institution calculates loss locally.
2. Backpropagation runs through the Tail.
3. The institution obtains the gradient with respect to the Backbone output.
4. That gradient is sent to the compute peer.
5. The compute peer backpropagates through the Backbone.
6. The compute peer updates the Backbone parameters.
7. The gradient with respect to the Backbone input is returned.
8. The institution backpropagates through the Head and updates its local Head parameters.

The implementation must correctly preserve the computation graph across the distributed boundary using detached tensors and explicitly exchanged gradients.

## 6. Tensor Exchange Contract

The following logical payloads are required.

### Activation payload

```python
{
    "institution_id": str,
    "batch_id": str,
    "activations": torch.Tensor,
    "backbone_version": str
}
```

### Backbone output

```python
{
    "institution_id": str,
    "batch_id": str,
    "activations": torch.Tensor,
    "backbone_version": str
}
```

### Gradient payload

```python
{
    "institution_id": str,
    "batch_id": str,
    "gradient": torch.Tensor,
    "backbone_version": str
}
```

These are logical payload definitions. The transport layer may serialize them into the format required by the communication implementation.

The exact serialization format must be agreed upon before network integration.

## 7. Checkpoint Interface

The coordinator owns the canonical Backbone checkpoint.

The compute peer must support:

- Loading a Backbone state dictionary.
- Checking the expected model version.
- Performing the assigned training computation.
- Returning the updated Backbone state dictionary.
- Identifying the version associated with the update.

Head and Tail checkpoints remain institution-specific.

Use standard PyTorch `state_dict` and `load_state_dict` wherever possible.

## 8. Responsibility Boundaries

### Split Learning Engine
Responsible for:
- Head, Backbone and Tail.
- Forward propagation.
- Backward propagation.
- Gradient exchange logic.
- Optimizers and model updates.
- Model initialization.
- Checkpoint loading and saving utilities.

### Coordinator
Responsible for:
- Assigning institutions and compute peers.
- Managing sequential training.
- Distributing canonical Backbone checkpoints.
- Receiving and storing updated Backbone checkpoints.
- Tracking training state.

### Dataset and Privacy
Responsible for:
- Dataset loading.
- Dataset partitioning.
- Data loaders.
- Activation noise configuration and injection.
- Activation perturbation metrics.

### Infrastructure
Responsible for:
- Docker and Docker Compose.
- Service startup and networking.
- Deployment configuration.
- Dashboard and monitoring.

## 9. Communication Boundary

The split-learning engine must not independently implement coordinator scheduling or Flower orchestration.

The engine should expose functions and classes that can be called by the coordinator and compute-peer services.

HTTP communication and serialization belong to the relevant service/infrastructure integration layer, unless explicitly agreed otherwise.

## 10. Important Constraints

- Raw input data stays at its institution.
- Labels stay at their institution.
- The Backbone is shared.
- Head and Tail weights are institution-specific.
- The compute peer must not receive labels.
- The engine must support separate processes and eventual network communication.
- Activation noise is applied outside the core model architecture.
- No formal differential privacy guarantee should be claimed merely because noise is added.

## 11. Compatibility

If a required interface is missing or ambiguous:
1. Identify the missing information.
2. Propose the smallest change.
3. Keep the implementation isolated until the change is agreed upon.

Do not silently change shared contracts.