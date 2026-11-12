# Split Learning Engine

U-shaped split learning in PyTorch (Member 2). The Head and Tail run at the
data-owning institution; the shared Backbone runs at the assigned compute peer.
Raw inputs and labels never leave the institution.

The engine does not handle transport. Every method takes and returns plain tensors and
strings, so the coordinator and peer services can wrap it in HTTP or Flower
calls without changing it.

## Architecture

```text
INSTITUTION (data owner)                       COMPUTE PEER
------------------------                       ------------
inputs ─> Head ─> activations ─(noise)───────> Backbone.forward
                                                    │
labels ─> Tail <─ backbone output <─────────────────┘
           │
         loss
           │ backward
           ├──> d loss / d backbone output ───> Backbone.backward, optimizer step
           │                                        │
Head.backward, optimizer step <─ d loss / d activations
```

| File | Contents |
| --- | --- |
| `config.py` | `SplitModelConfig`: input shape, widths, split point, classes, seed |
| `path_cnn.py` | `path_cnn_v1_config()`: the MedWeavePathCNN-v1 preset for PathMNIST |
| `head.py`, `backbone.py`, `tail.py` | The three `nn.Module`s and their `build_*` factories |
| `layers.py` | The conv block shared by Head and Backbone |
| `forward.py`, `backward.py` | Single-party pass functions (stateless) |
| `institution.py` | `Institution`, `create_institutions`, `StepMetrics` |
| `compute_peer.py` | `ComputePeer`, `BackboneUpdate` |
| `checkpoint.py` | Backbone and institution checkpoint I/O, NumPy weights for Flower |
| `training.py` | `train_step`, `train_batches`, `evaluate`, `summarize` |
| `validation.py`, `errors.py` | Boundary checks and error types |

### Configuring the model

The engine itself is not tied to a dataset. The PathMNIST model is a preset
(see [MedWeavePathCNN-v1](#medweavepathcnn-v1-pathmnist)). Every field:

```python
SplitModelConfig(
    in_channels=1,                 # 3 for RGB
    image_size=(28, 28),           # used to validate tensors at every boundary
    head_channels=(16,),           # one conv block (conv3x3, [BN], ReLU, maxpool2) per entry
    backbone_channels=(32, 32),
    tail_hidden=(),                # hidden Linear layers in the Tail
    num_classes=2,
    seed=0,                        # same seed => same initial weights everywhere
    batch_norm=False,              # BatchNorm2d after every conv
    backbone_final_pool=True,      # False: last Backbone block keeps its spatial size
    tail_dropout=0.0,              # Dropout before the Tail's classifier, in [0, 1)
)
```

Moving an entry from `backbone_channels` to `head_channels` moves the split
point deeper. `config.head_output_shape` and `config.backbone_output_shape`
give the per-sample shapes that cross each boundary. An invalid configuration,
such as an image too small for the number of pooling blocks, raises
`ValueError` when the config is created.

The last three fields only add layers when they are switched on, so the
default placeholder CNN keeps the same `state_dict` keys as before they were
introduced.

## MedWeavePathCNN-v1 (PathMNIST)

The model for the PathMNIST experiment: RGB 28×28 histology patches, 9 classes,
trained with `CrossEntropyLoss`. It has no pretrained weights and about 95k
parameters.

```text
                     INSTITUTION                COMPUTE PEER                 INSTITUTION
input [B,3,28,28]
  │
  ├─ HEAD (stage 1) ─────────┐
  │  Conv 3→32, BN, ReLU,    │
  │  MaxPool                 │
  │  → [B,32,14,14] ─────────┼──> BACKBONE (stages 2+3)
  │                          │    Conv 32→64, BN, ReLU, MaxPool  → [B,64,7,7]
  │                          │    Conv 64→128, BN, ReLU          → [B,128,7,7] ──> TAIL
  │                          │                                                     AvgPool(1), Flatten
  │                          │                                                     Dropout(0.2)
  │                          │                                                     Linear 128→9
  │                          │                                                     → [B,9] logits
```

| Component | Layers | Output | Parameters |
| --- | --- | --- | --- |
| Head | Conv2d(3, 32, 3, padding=1), BatchNorm2d(32), ReLU, MaxPool2d(2) | `[B, 32, 14, 14]` | 960 |
| Backbone | Conv2d(32, 64, 3, padding=1), BatchNorm2d(64), ReLU, MaxPool2d(2) | `[B, 64, 7, 7]` | |
| | Conv2d(64, 128, 3, padding=1), BatchNorm2d(128), ReLU | `[B, 128, 7, 7]` | 92,736 |
| Tail | AdaptiveAvgPool2d(1), Flatten, Dropout(0.2), Linear(128, 9) | `[B, 9]` | 1,161 |

The Tail returns raw logits. There is no softmax in the model, because
`F.cross_entropy` applies log-softmax itself.

### Four institutions and one shared Backbone

```python
from split_learning.backbone import build_backbone
from split_learning.compute_peer import ComputePeer
from split_learning.institution import create_institutions
from split_learning.path_cnn import path_cnn_v1_config

config = path_cnn_v1_config()      # default split: stage 1 | stages 2+3

institutions = create_institutions(["inst-1", "inst-2", "inst-3", "inst-4"], config)
canonical = build_backbone(config).state_dict()   # coordinator's initial checkpoint

peer = ComputePeer(config)                        # any institution acting as compute peer
peer.load_backbone(canonical, "0")
```

Each institution gets its own Head, Tail and Head/Tail optimizers. They start
from identical weights (same `seed`), but no parameter or BatchNorm buffer is
shared. The rest of the workflow is unchanged from the generic engine (see
[Training and evaluation utilities](#training-and-evaluation-utilities)).

**BatchNorm and checkpoints.** BatchNorm keeps `running_mean`, `running_var`
and `num_batches_tracked` as buffers. They are part of `state_dict()`, so they
travel in Backbone checkpoints and through `get_weights`/`weights_to_state`
together with the weights. Each institution's Head statistics stay local, like
its weights. `train_batches` puts modules in train mode (batch statistics,
dropout on). `evaluate` uses eval mode (running statistics, dropout off).
BatchNorm2d also works with a batch of one, because it normalises over
`B×H×W`.

### Changing the split point

```python
config = path_cnn_v1_config(head_stages=2)   # Head: stages 1+2, Backbone: stage 3
config.head_output_shape                     # (64, 7, 7)
config.backbone_output_shape                 # (128, 7, 7), unchanged
```

`head_stages` can be 1 (default) or 2. Stage 3 never pools, and it is always
the last Backbone block, so the Backbone output shape and the Tail stay the
same at either split point. For widths or stages beyond these two options,
build a `SplitModelConfig` directly with the same `batch_norm`,
`backbone_final_pool` and `tail_dropout` settings.

A Backbone checkpoint only fits the split point it was trained with. Loading a
`head_stages=1` checkpoint into a `head_stages=2` peer raises
`CheckpointError`. Every participant must therefore use the same config.

## How the forward pass works

1. `Institution.head_forward(batch_id, inputs)` runs the Head and returns a
   **detached copy** of the activations. The Head's graph stays with the
   institution until the gradient comes back.
2. Noise from the privacy layer is applied to that copy, outside the model.
3. `ComputePeer.forward(batch_id, activations, backbone_version)` checks the
   version and shape, turns the activations into a fresh leaf tensor
   (`detach().requires_grad_()`), runs the Backbone and returns a detached copy
   of the output.
4. `Institution.tail_step(batch_id, backbone_output, labels)` runs the Tail and
   computes cross-entropy against the local labels.

## How gradient exchange works

Each boundary tensor is a leaf on the receiving side, so after `backward()`
its `.grad` holds exactly the gradient that must travel back:

1. `tail_step` back-propagates through the Tail, steps the Tail optimizer and
   returns `d loss / d backbone_output` in `TailResult.gradient`.
2. `ComputePeer.backward(batch_id, gradient, backbone_version)` back-propagates
   that gradient through the Backbone, steps the Backbone optimizer and returns
   `d loss / d activations`.
3. `Institution.head_backward(batch_id, gradient)` applies it to the Head output
   and steps the Head optimizer.

Gradients at both boundaries are tested to match a monolithic Head, Backbone and Tail
model to within float tolerance.

Every component has its own optimizer, and each backward step zeroes that
optimizer's gradients first, so batches never accumulate into each other.

**Noise during backprop:** the gradient is computed with respect to the
*noisy* activations and applied unchanged to the clean Head output
(straight-through). This is exact for additive noise because
`d(a + n)/da = I`; for non-additive noise it is an approximation.

**Batch state:** each party holds at most one batch between forward and
backward (`PendingBatch`). Calls must use the same `batch_id`, and a second
forward before the backward raises `ProtocolError`. Finishing or discarding a
batch drops its graph, and a test confirms the memory is freed.

## Creating institutions

```python
from split_learning.institution import create_institutions

institutions = create_institutions(["inst-1", "inst-2", "inst-3", "inst-4"], config)
```

Each institution gets its own Head, Tail, Head optimizer and Tail optimizer.
They start with identical weights and share no storage. Pass
`optimizer_factory` to choose the optimizer, e.g.
`lambda params: torch.optim.Adam(params, lr=1e-3)`. The default is SGD with
`lr=0.01`.

## Using a compute peer

```python
from split_learning.compute_peer import ComputePeer

peer = ComputePeer(config)
peer.load_backbone(state_dict, version="3")          # from the coordinator

output = peer.forward(batch_id, activations, "3")    # per batch
input_gradient = peer.backward(batch_id, gradient, "3")

update = peer.backbone_update()   # BackboneUpdate(base_version="3", state_dict, num_batches)
```

- A peer must load a Backbone before doing anything else. Loading resets the
  optimizer, so optimizer state is not carried between turns.
- A payload with a different `backbone_version` raises `ProtocolError`.
- `backbone_update()` returns a copy of the weights. It refuses while a batch
  is still unfinished.
- `discard_pending()` drops an unfinished batch, e.g. after a disconnect.
- `infer(activations, version)` runs the Backbone for evaluation with no graph.

## Checkpoints

```python
from split_learning.checkpoint import (
    save_backbone_checkpoint, read_backbone_checkpoint,
    save_institution_checkpoint, load_institution_checkpoint,
    get_weights, weights_to_state,
)

save_backbone_checkpoint("backbone_v4.pt", update.state_dict, version="4")
peer.load_backbone(*read_backbone_checkpoint("backbone_v4.pt"))

save_institution_checkpoint("inst-1.pt", institutions["inst-1"])   # Head, Tail, both optimizers
load_institution_checkpoint("inst-1.pt", institutions["inst-1"])

arrays = get_weights(peer.backbone)                                 # Flower NDArrays
peer.load_backbone(weights_to_state(peer.backbone, arrays), "4")
```

Files are read with `torch.load(weights_only=True)`. A load is all or nothing:
a mismatched architecture, a corrupt file or another institution's checkpoint
raises `CheckpointError` and leaves the model unchanged.

## Training and evaluation utilities

`training.py` runs every party in one process for simulation and tests:

```python
import torch

from split_learning.backbone import build_backbone
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.institution import create_institutions
from split_learning.training import evaluate, train_batches

config = SplitModelConfig(in_channels=1, image_size=(28, 28), num_classes=2)
institutions = create_institutions(["inst-1", "inst-2", "inst-3", "inst-4"], config)
peer = ComputePeer(config)

canonical, version = build_backbone(config).state_dict(), 0   # coordinator's job
batches = [(torch.randn(8, 1, 28, 28), torch.randint(0, 2, (8,))) for _ in range(5)]

peer.load_backbone(canonical, str(version))
summary = train_batches(institutions["inst-1"], peer, batches)
print(f"loss={summary.loss:.3f} accuracy={summary.accuracy:.2f}")

update = peer.backbone_update()          # -> coordinator stores as version 1
print(evaluate(institutions["inst-1"], peer, batches))
```

- `train_step(institution, peer, inputs, labels, batch_id, perturb)` runs one
  batch and returns `StepMetrics(loss, correct, total)`. If any step fails,
  both parties discard the batch.
- `train_batches(...)` puts both parties in train mode and returns a
  `Summary(loss, accuracy, num_batches, num_samples)` weighted by sample count.
- `evaluate(...)` runs in eval mode under `no_grad`, updates nothing and
  restores the previous modes.
- `perturb` is the noise hook: `Callable[[Tensor], Tensor]`, applied to the
  detached activations.

## Input contract

| Tensor | Shape | dtype |
| --- | --- | --- |
| inputs | `(B, *config.input_shape)` | float |
| labels | `(B,)`, values in `[0, num_classes)` | `torch.long` |
| activations, gradient to Head | `(B, *config.head_output_shape)` | float |
| backbone output, gradient to peer | `(B, *config.backbone_output_shape)` | float |

A violation raises `ShapeError`, which is also a `ValueError`. Out-of-order
calls, wrong batch ids and wrong versions raise `ProtocolError`. All engine
errors derive from `SplitLearningError`.

## Running the tests

```bash
pip install -r split_learning/requirements.txt
python -m pytest tests/unit/split_learning -v
```

Run from the repository root (`python -m` puts it on `sys.path`). All tests
use synthetic data.

## Interface notes for the team

These need agreement and have **not** been changed in the shared documents:

1. **File names.** `SPEC.md` and the global rules refer to `GLOBAL_RULES.md`
   and `INTERFACES.md`, but the files are `global_rules.md` and
   `shared_interfaces.md`, and they are not committed yet.
2. **Version type.** `shared_interfaces.md` has `backbone_version: str`, while
   `SPEC.md` §3.2 stores `model_checkpoints.version` as `INTEGER`. The engine
   treats the version as an opaque string. Proposal: the coordinator keeps the
   integer and sends `str(version)`.
3. **`institution_id` in payloads** is not used by the engine. The relay or
   peer service should check it against the coordinator's assignment.
4. **Dataset contract (Member 3):** loaders should yield float
   `(B, C, H, W)` inputs and `torch.long` `(B,)` labels, as in the table above.
5. **Noise hook (Member 3):** `Callable[[Tensor], Tensor]`, returning a tensor
   of the same shape.
6. **Model architecture.** `SPEC.md` §2.8 and §8 list the dataset and
   architecture as deferred. MedWeavePathCNN-v1 assumes PathMNIST has now been
   chosen. It lives in its own preset (`path_cnn.py`), and the engine remains
   dataset-agnostic. The default `SplitModelConfig()` is still the placeholder
   CNN. Services should build their config with `path_cnn_v1_config()`.
7. **PathMNIST loader (Member 3):** the model expects float `[B, 3, 28, 28]`
   inputs and `torch.long` labels in `[0, 9)`. MedMNIST provides labels as
   `[B, 1]`, so the loader needs to squeeze them to `[B]`.

## Known limitations

- CPU only. Device placement is not handled yet.
- One batch in flight per party, which matches sequential training.
- Backbone optimizer state is reset at every turn; only weights are
  checkpointed.
- `SplitModelConfig()` defaults to a small placeholder CNN. PathMNIST uses
  `path_cnn_v1_config()`.
- In MedWeavePathCNN-v1 each conv keeps its bias, as specified. A bias directly
  before BatchNorm is cancelled by the mean subtraction and receives no
  gradient. That costs 224 unused parameters. Setting `bias=False` would remove
  them, but it would change the checkpoint keys.
- BatchNorm statistics in the Backbone are updated by whichever institution's
  data passes through it, so they are shared like the weights. With strongly
  non-IID partitions this may need revisiting (e.g. GroupNorm).
- Gradients sent to the peer are derived from the labels, a known split-learning
  leakage channel. The noise is a perturbation, not a formal privacy guarantee
  (SPEC §1.7).
