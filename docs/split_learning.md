# Split Learning Engine (Member 2)

U-shaped split learning in PyTorch. Head and Tail run at the data-owning
institution; the shared Backbone runs at the assigned compute peer.

```text
institution                         compute peer
-----------                         ------------
head_forward(x)      --smashed-->   backbone_forward
                                          |
tail_forward(y)      <--output---         |
tail_backward        --grad_out-->  backbone_backward  (Backbone updated)
head_backward        <--grad_smashed--
```

## Modules

| File | Contents |
| --- | --- |
| `config.py` | `SplitModelConfig` (channels, classes, seed) |
| `head.py`, `backbone.py`, `tail.py` | The three `nn.Module`s and `build_*` factories |
| `forward.py` | `head_forward`, `backbone_forward`, `tail_forward` |
| `backward.py` | `tail_backward`, `backbone_backward`, `head_backward` |
| `checkpoint.py` | `save_checkpoint`, `load_checkpoint`, `get_weights`, `set_weights` |
| `step.py` | `train_step` running all three parties in one process |

The default layers are a placeholder small CNN. The dataset and task are not
chosen yet (SPEC §2.8), so the shapes come from `SplitModelConfig`. The Tail
pools adaptively, so any input image size works.

## Usage

### Institution (data owner)

```python
from split_learning.config import SplitModelConfig
from split_learning.head import build_head
from split_learning.tail import build_tail
from split_learning.forward import head_forward, tail_forward
from split_learning.backward import tail_backward, head_backward

config = SplitModelConfig(in_channels=1, num_classes=2)
head, tail = build_head(config), build_tail(config)
head_opt = torch.optim.SGD(head.parameters(), lr=0.01)
tail_opt = torch.optim.SGD(tail.parameters(), lr=0.01)

smashed, head_pass = head_forward(head, images)
smashed = perturb(smashed)                  # Member 3's noise, optional
# send smashed -> peer, receive backbone_output
tail_pass = tail_forward(tail, backbone_output, labels)
grad_output = tail_backward(tail_pass, tail_opt)
# send grad_output -> peer, receive grad_smashed
head_backward(head_pass, grad_smashed, head_opt)
```

### Compute peer

```python
from split_learning.backbone import build_backbone
from split_learning.checkpoint import set_weights, get_weights
from split_learning.forward import backbone_forward
from split_learning.backward import backbone_backward

backbone = build_backbone(config)
set_weights(backbone, weights_from_coordinator)
optimizer = torch.optim.SGD(backbone.parameters(), lr=0.01)

backbone_output, backbone_pass = backbone_forward(backbone, smashed)
# return backbone_output, then receive grad_output
grad_smashed = backbone_backward(backbone_pass, grad_output, optimizer)
# return grad_smashed; at the end of the turn: get_weights(backbone) -> coordinator
```

### Single process (simulation and tests)

```python
from split_learning.step import Institution, ComputePeer, train_step

loss = train_step(institution, peer, images, labels, perturb=noise_fn)
```

## Contract

- Every tensor that crosses the network is **detached**, so no autograd state
  leaves a party. Shapes are `(B, head_channels, H/2, W/2)` for `smashed` and
  `grad_smashed`, and `(B, backbone_channels, H/4, W/4)` for `backbone_output`
  and `grad_output`.
- The `*Pass` objects hold the local graph between forward and backward. Each
  party keeps them in memory and never sends them.
- Each component has its own optimizer. A combined Head+Tail optimizer would
  move the Head when `tail_backward` steps.
- **Noise during backprop (SPEC §2.2):** noise is applied to the detached
  activations, outside every graph. The peer's gradient is with respect to the
  noisy activations and is applied unchanged to the clean Head output
  (straight-through). For additive noise this is exact. For non-additive noise
  it is an approximation.
- Identical `SplitModelConfig.seed` gives identical initial weights across
  institutions (SPEC §2.3). Building a model does not change the global RNG.
- Checkpoints: `.pt` files hold `{"version", "state_dict"}` and load with
  `weights_only=True`. `get_weights`/`set_weights` use a list of NumPy arrays in
  `state_dict` order, which is Flower's `NDArrays` format.

## Proposed additions to `INTERFACES.md`

`INTERFACES.md` and `GLOBAL_RULES.md` are not in the repository yet, so these
items still need team agreement:

1. **Exchange messages** (institution ↔ coordinator relay ↔ peer):
   `activations`, `backbone_output`, `grad_output`, `grad_smashed`, each tagged
   with `session_id`, `round_number`, `data_owner_id` and a per-batch `step_id`.
   A peer can then match a gradient to its pending `BackbonePass`.
2. **Tensor wire format** for HTTP bodies. One option is `torch.save` to bytes
   with `torch.load(weights_only=True)` on receipt; another is raw NumPy plus
   shape and dtype. Not implemented here until the team decides.
3. **Noise hook signature:** `Callable[[Tensor], Tensor]`, applied to detached
   activations. Member 3 should confirm this.
4. **Backbone transfer:** `get_weights`/`set_weights` for Flower, with
   `save_checkpoint`/`load_checkpoint` for the coordinator's `storage_path`.
5. **Optimizer and hyperparameters** are created by the caller. The peer's
   Backbone optimizer lasts for one training turn. Momentum and Adam state are
   not carried between turns because only weights are checkpointed.

## Dependencies

`torch` (developed against 2.9) and `numpy`. Tests use `pytest`.

## Tests

```bash
python -m pytest tests/unit/split_learning
```

The tests cover forward shapes and detachment, gradient equality with an
equivalent monolithic model, gradient shapes at each exchange, straight-through
noise, updates to all three components, checkpoint and weight round-trips, loss
decreasing over steps, and a two-institution turn handoff through a Backbone
checkpoint.
