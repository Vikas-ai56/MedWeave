# Codebase map

Dependencies point one way: `split_learning`, `institution_data`, `privacy` and `tracking` depend on
nothing else in the repo; `coordinator` uses all four; `experiments` sits on top of everything.

| Path | Owner | What it holds |
|---|---|---|
| `split_learning/` | Member 2 | U-shaped engine: Head/Backbone/Tail, forward/backward passes, `Institution`, `ComputePeer`, checkpoints (institution checkpoints record their optimizer type), `optimizer_factory`, `train_batches`/`evaluate` (optional evaluation noise). See `split_learning/README.md`. |
| `institution_data/` | Member 3 | PathMNIST source (`pathmnist.py`), stratified partitioning and manifest (`partition.py`), provisioning and opening shards (`provision.py`), tensors and per-turn DataLoaders (`batches.py`), `python -m institution_data prepare` (`__main__.py`). |
| `privacy/` | Member 3 | Gaussian activation noise and `perturb` hook (`noise.py`), perturbation statistics and heatmaps (`statistics.py`). |
| `coordinator/scheduling/` | Member 1 | Turn order and peer rotation. |
| `coordinator/checkpoint_manager/` | Member 1 | Versioned canonical Backbone store. |
| `coordinator/relay/` | Member 1 | Per-turn relay rules and HTTP transport for activations and gradients. |
| `coordinator/flower_app/` | Member 1 | Flower ServerApp/ClientApp, `protocol.py` (`MODEL_CONFIG`: 3×28×28, 9 classes; message types incl. `evaluate.held_out`), `session.py` (turns, then per-round local evaluation; returns `SessionResult`), `roles.py`, `pyproject.toml` run settings, `run_local.sh` (provisions data, exits on the SuperLink's run status). The ClientApp's owner turn reads its shard lazily and applies the noise hook; both roles build optimizers from the `optimizer` setting. |
| `tracking/` | shared | Experiment tracking: `Tracker` interface and no-op, MLflow backend (only importer of `mlflow`), store location, `python -m tracking ui`. See `docs/features/experiment-tracking.md`. |
| `experiments/` | Member 3 | `noise_comparison.py`: in-process noise comparison (clean or noisy evaluation) and activation inspection; results in `experiments/results/`. |
| `tests/conftest.py` | shared | Tiny official-format PathMNIST fixture, provisioned through the production code. |
| `tests/unit/<package>/` | per package | Unit tests; `tests/integration/` drives data and noise through the real engine. |
| `docs/` | shared | `coordinator_guide.md` (running sessions), `features/data-and-privacy.md`, `DECISIONS.md`, this map. |

Local, git-ignored: `venv/`, `.medweave_local/` (session state), `.medweave_verify/` (verification
runs). Outside the repo: `~/.medweave/data` (dataset, or `$MEDWEAVE_DATA_DIR`) and `~/.medweave/mlflow`
(tracking store, or `$MEDWEAVE_TRACKING_DIR`).
