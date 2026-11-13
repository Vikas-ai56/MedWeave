# MedWeave

**Privacy-Preserving Collaborative AI Framework**

MedWeave is a collaborative AI framework that allows multiple institutions to train a shared machine learning model without sharing their raw datasets.

The project explores a combination of U-shaped Split Learning and collaborative training, where institutions retain their local data while using computing resources from other participating institutions.

## Overview

Sharing data between institutions can be difficult due to privacy concerns, institutional policies, and limited computing resources.

MedWeave aims to address this by separating data ownership from model computation. Each institution keeps its dataset locally and participates in training through an exchange of intermediate model representations.

The initial implementation uses four simulated institutions running on a single machine. The system is designed to be extended to multiple machines connected over a local network.

## How It Works

1. Each institution maintains its own local dataset.
2. A local model processes the input data and generates intermediate activations.
3. The activations are sent to an assigned compute peer for further processing.
4. The resulting gradients are returned to the institution.
5. Institutions take turns contributing to a shared Backbone model.
6. The coordinator manages training and maintains the latest shared model checkpoint.

Raw images and labels remain with their respective institutions.

## Architecture

MedWeave consists of four main components:

- **Coordinator:** Manages institutions, training schedules, compute assignments, and shared model checkpoints.
- **Institutional Clients:** Hold local datasets and execute the institution-specific parts of the model.
- **Compute Peers:** Perform the remote Backbone computation assigned to them.
- **Privacy Layer:** Supports configurable noise injection into intermediate activations before transmission.

## Technology Stack

| Component              | Technology             |
| ---------------------- | ---------------------- |
| Programming Language   | Python                 |
| Deep Learning          | PyTorch                |
| Training Orchestration | Flower                 |
| Backend API            | FastAPI                |
| Frontend               | React                  |
| Database               | SQLite                 |
| Containerization       | Docker, Docker Compose |
| Version Control        | Git, GitHub            |

## Project Structure

```text
MedWeave/
├── coordinator/        # Training coordination and model management
├── institutional_client/ # Local institutional training logic
├── split_learning/     # Head, Backbone and Tail implementation
├── institution_data/  # PathMNIST loading, institutional partitions, DataLoaders
├── privacy/            # Activation noise and perturbation statistics
├── experiments/        # Noise comparison runner and results
├── tracking/           # Experiment tracking (MLflow behind a small interface)
├── dashboard/          # Monitoring interface
├── tests/              # Unit and integration tests
├── docker/             # Container configuration
├── docs/               # Architecture and project documentation
├── GLOBAL_RULES.md
├── SPEC.md
├── INTERFACES.md
└── README.md
```

*The structure may evolve as implementation progresses.*

## Current Development Scope

The initial milestone focuses on a working end-to-end prototype:

- Four simulated institutions.
- Local datasets with separate data loaders.
- U-shaped Split Learning implementation.
- Coordinator-managed sequential training.
- Rotating compute peer assignments.
- Configurable activation noise.
- A shared Backbone checkpoint maintained by the coordinator.

The prototype will initially run on a single machine using Docker Compose, with multi-machine execution planned as a subsequent milestone.

## Getting Started

### Prerequisites

- Docker and Docker Compose, **or** Python 3.14 and Node.js (for the dashboard)
- Git

### Clone the Repository

```bash
git clone <repository-url>
cd MedWeave
```

### Run the Project

Two ways to run a full session (4 institutions, PathMNIST, activation noise, TLS) on one machine.
Both download PathMNIST (205 MB) on first use.

**With Docker** (see [`docs/docker.md`](docs/docker.md)):

```bash
docker compose up -d --build        # coordinator, 4 institutions, dashboard
docker compose run --rm session     # one training session (3 rounds by default)
open http://127.0.0.1:8080          # live dashboard
```

**With a Python virtual environment** (macOS or Linux):

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
coordinator/flower_app/run_local.sh           # terminal 1: a training session
dashboard/run_dashboard.sh                    # terminal 2: dashboard on http://127.0.0.1:8080 (needs Node.js)
python -m tracking ui                         # optional: MLflow run history
```

Settings (rounds, noise, learning rate, ...) are in `coordinator/flower_app/pyproject.toml`; override one
for a run with `--run-config "noise-std=0.25"`. The teammate guide,
[`docs/coordinator_guide.md`](docs/coordinator_guide.md), covers the details and the multi-laptop setup.

## Team

Developed as an academic project by a team of four students at **RV College of Engineering, Bengaluru**.

---

*MedWeave is an academic proof-of-concept. It is not intended for clinical use and does not currently provide a formally verified differential privacy guarantee.*
