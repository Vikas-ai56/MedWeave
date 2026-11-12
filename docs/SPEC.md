# MedWeave — Build Specification

**Version:** 1.0  
**Project type:** Academic research proof of concept  
**Primary deliverable:** Working collaborative AI software prototype  
**Team size:** 4 members  
**Initial development target:** 1–2 weeks for the MVP

---

# 1. Product Requirements and Functional Scope

## 1.1 Product overview

MedWeave is a privacy-oriented collaborative AI framework that enables multiple institutions to jointly train AI models without centralizing their underlying datasets.

It combines U-shaped split learning, rotating peer computation and noise perturbation of intermediate activations to explore collaborative model development while retaining local control over training data.

The initial implementation will simulate four institutions using four independent clients running on one laptop. The framework will subsequently be deployed across four physical laptops.

**The central objective is to get the complete collaborative training workflow working.**

The project is a technical proof of concept. It is not a production healthcare platform or a clinically validated medical AI system.

## 1.2 Problem statement

Medical AI development faces barriers to collaboration because institutions may have restrictions on sharing medical datasets and may have unequal access to computational resources.

MedWeave explores a collaborative learning framework in which institutions retain their local datasets while using computation provided by other participating institutions to train a shared model.

The framework aims to demonstrate collaborative model development without requiring centralization of the underlying training datasets.

## 1.3 Intended users

| Stakeholder                | Role                                                         |
| -------------------------- | ------------------------------------------------------------ |
| Healthcare institutions    | Motivating institutional setting and potential beneficiaries |
| Medical AI researchers     | Intended users of the collaborative learning framework       |
| Participating institutions | Own local datasets and contribute to training                |
| Compute peers              | Execute Backbone computation for another institution         |
| Coordinator                | Orchestrates training and maintains shared model state       |

All institutions in the initial PoC are simulated. No real hospital partnership or patient-data deployment is assumed.

## 1.4 Core functional requirements

| ID    | Requirement                                                              | Priority                 |
| ----- | ------------------------------------------------------------------------ | ------------------------ |
| FR-01 | Support four independent institutional clients                           | Mandatory                |
| FR-02 | Keep each institution's raw training dataset local                       | Mandatory                |
| FR-03 | Implement U-shaped split learning                                        | Mandatory                |
| FR-04 | Maintain one shared Backbone model                                       | Mandatory                |
| FR-05 | Assign a different compute peer to each data-owning institution          | Mandatory                |
| FR-06 | Rotate compute assignments between institutional training turns          | Mandatory                |
| FR-07 | Perturb smashed activations with configurable noise                      | Mandatory                |
| FR-08 | Coordinate sequential institutional training                             | Mandatory                |
| FR-09 | Store and distribute the latest shared Backbone checkpoint               | Mandatory                |
| FR-10 | Complete a collaborative training run across four simulated institutions | MVP acceptance criterion |
| FR-11 | Support a basic classification inference demonstration                   | Later milestone          |
| FR-12 | Visualize collaboration and activation perturbation                      | Later milestone          |
| FR-13 | Support institutional disconnection and reconnection                     | Later enhancement        |

## 1.5 Core user workflow

1. Four institutions register with the coordinator.
2. Each institution loads its own local partition of a public dataset.
3. The coordinator initializes the shared Backbone.
4. The coordinator selects the next data-owning institution.
5. It assigns a different institution as the compute peer.
6. The data-owning institution processes local samples through its Head.
7. Intermediate activations are perturbed with noise.
8. The activations are sent through the coordinator to the assigned compute peer.
9. The compute peer executes the Backbone.
10. The output returns to the data-owning institution.
11. Its local Tail calculates the loss using locally retained labels.
12. Gradients propagate backwards through the split model.
13. The compute peer updates the Backbone.
14. The updated Backbone checkpoint returns to the coordinator.
15. The next institution receives its training turn.
16. The process continues until all four institutions have completed their contributions.

The coordinator retains the latest shared Backbone. Institutional Heads and Tails remain local.

## 1.6 Non-functional requirements

- Reproducible development and deployment.
- Modular Python implementation.
- Consistent interfaces between independently developed modules.
- Docker-based environment consistency.
- Secure network communication using TLS.
- Basic error handling and training-status logging.
- Ability to simulate all four institutions on one laptop.
- Ability to move the same system to four physical laptops.

## 1.7 Explicitly excluded from the initial scope

- Production hospital deployment.
- PACS/EMR integration.
- Real patient data.
- Clinical validation or diagnostic claims.
- Formal differential privacy guarantees.
- Membership inference and reconstruction attack experiments.
- Blockchain governance and Proof of Useful Learning.
- Reputation mechanisms.
- Kubernetes or large-scale cloud deployment.
- Complex fault-tolerance infrastructure.
- Extensive benchmarking.

Noise injection is an experimental privacy-oriented mechanism. MedWeave must not claim formal differential privacy or proven protection against all information leakage.

---

# 2. Technical Architecture and Technology Stack

## 2.1 Architectural overview

MedWeave uses a decentralized peer-computation arrangement with a lightweight centralized coordinator.

The coordinator is trusted. Compute peers are treated as honest-but-curious: they execute the prescribed computation but may inspect the activations they receive.

The architecture is not coordinator-free.

### Main components

| Component             | Responsibility                                      | Technology                   |
| --------------------- | --------------------------------------------------- | ---------------------------- |
| Institutional client  | Local dataset, Head, Tail and training state        | Python, PyTorch              |
| Compute peer          | Remote Backbone computation                         | Python, PyTorch              |
| Coordinator           | Scheduling, Backbone state and communication relay  | Flower + custom Python logic |
| Dashboard backend     | Expose system status and visualization data         | Proposed: FastAPI            |
| Web dashboard         | Display collaboration and activation visualizations | Proposed: React              |
| Container environment | Run simulated institutions                          | Docker, Docker Compose       |
| Metadata storage      | Store training and coordination metadata            | Proposed: SQLite             |

## 2.2 U-shaped split-learning architecture

Each institution maintains two local model components:

- **Head:** Processes the original input and produces intermediate activations.
- **Tail:** Receives the output from the shared Backbone, calculates predictions and computes the loss using local labels.

The remote compute peer executes the shared Backbone.

Conceptually:

```text
INSTITUTION A
  Local Image
      |
      v
  Local Head
      |
      v
  Intermediate Activations
      |
      v
  Noise Injection
      |
      v
  COORDINATOR (trusted relay)
      |
      v
  INSTITUTION B (compute peer)
      |
      v
  Shared Backbone
      |
      v
  Backbone Output
      |
      v
  COORDINATOR
      |
      v
  INSTITUTION A
      |
      v
  Local Tail
      |
      v
  Loss Calculation using Local Labels
```

The backward pass propagates gradients through the Tail, Backbone and Head.

The compute peer updates the Backbone, while the data-owning institution updates its local Head and Tail.

The exact treatment of the activation-noise operation during backpropagation must be specified in the split-learning implementation.

## 2.3 Shared Backbone management

There is one canonical Backbone checkpoint.

The coordinator:

1. Initializes the Backbone.
2. Distributes its latest checkpoint to the assigned compute peer.
3. Receives the updated checkpoint after the institution's training turn.
4. Stores the updated version.
5. Distributes the latest version for the next turn.

Each institution has its own Head and Tail. Their architectures are identical, their initial weights are shared, and their subsequent weights evolve independently.

## 2.4 Peer-computation model

Training is sequential.

For four institutions:

| Training turn | Data-owning institution | Compute peer  |
| ------------- | ----------------------- | ------------- |
| 1             | Institution 1           | Institution 2 |
| 2             | Institution 2           | Institution 3 |
| 3             | Institution 3           | Institution 4 |
| 4             | Institution 4           | Institution 1 |

This is an illustrative rotation. The coordinator implements the assignment algorithm and excludes the data-owning institution from its own compute assignment.

One compute peer remains assigned throughout an institution's training turn.

Each institution completes its configured local training schedule before the next institution begins.

The framework should support configurable training rounds, although the initial live demonstration can use one short round.

## 2.5 Flower and PyTorch responsibilities

**Flower**

- Client registration.
- Coordination and scheduling.
- Sequential training orchestration.
- Backbone checkpoint distribution.
- Collection of updated Backbone checkpoints.

**PyTorch**

- Model architecture.
- Local Head and Tail.
- Shared Backbone.
- Split forward pass.
- Split backward pass.
- Gradient computation.
- Optimizer operations.
- Activation noise injection.

Custom Flower strategies and application logic will be required. Standard federated-learning workflows should not be assumed to implement U-shaped split learning automatically.

## 2.6 Communication architecture

The current design uses two complementary communication responsibilities:

- Flower manages client orchestration and shared Backbone state.
- Custom HTTP APIs handle the split-learning activation and gradient exchange, coordinated through the trusted relay.

This separation should be reflected in `INTERFACES.md` before parallel implementation.

Communication must use TLS when deployed across laptops.

Raw datasets and labels must not be transmitted through the coordinator.

The coordinator relays perturbed activations but does not persist or analyse them.

## 2.7 Activation noise

Noise is added to intermediate activations before they leave the data-owning institution.

The dashboard, in a later phase, should display:

1. Original medical image.
2. Activation representation before noise.
3. Activation representation after noise.
4. Relative activation change.

Proposed visualization metric:

\[
\text{Relative Change (\%)} =
\frac{\|A*{\text{noisy}}-A*{\text{original}}\|_2}
{\|A_{\text{original}}\|\_2}\times100
\]

This measures perturbation, not privacy protection.

The initial implementation does not claim a formal differential privacy guarantee.

## 2.8 Dataset and model strategy

The dataset and classification task have not yet been finalized.

Agreed constraints:

- Use a publicly available dataset.
- Divide it into four disjoint IID partitions.
- Simulate four institutions.
- Keep each institution's training partition local.
- Reserve a held-out test dataset at the coordinator for later evaluation.
- Select the model architecture only after selecting the dataset and classification task.
- Use one primary classification task, with optional additional tests later.

No particular dataset or classification architecture should be hardcoded into the framework prematurely.

---

# 3. Database Schema and Module Boundaries

## 3.1 Database philosophy

The database is for coordination and observability, not for centralizing medical data.

The coordinator must not store:

- Original training images.
- Patient labels.
- Raw institutional datasets.
- Intermediate activation tensors.
- Patient-identifying information.

The database should store only the metadata required to manage collaborative training.

**Proposed initial database:** SQLite.

This is a pragmatic choice for a single-machine PoC. It can be replaced with PostgreSQL later if deployment requirements justify it.

## 3.2 Proposed schema

### Table: `institutions`

| Field          | Type      | Description                        |
| -------------- | --------- | ---------------------------------- |
| institution_id | TEXT / PK | Unique institution identifier      |
| display_name   | TEXT      | Institution display name           |
| client_address | TEXT      | Client endpoint or service address |
| status         | TEXT      | Online, offline, training          |
| registered_at  | DATETIME  | Registration timestamp             |
| last_seen_at   | DATETIME  | Last heartbeat                     |

### Table: `training_sessions`

| Field                  | Type      | Description                         |
| ---------------------- | --------- | ----------------------------------- |
| session_id             | TEXT / PK | Unique training session             |
| status                 | TEXT      | Pending, running, completed, failed |
| current_round          | INTEGER   | Current collaborative round         |
| current_institution_id | TEXT / FK | Current data owner                  |
| created_at             | DATETIME  | Session creation time               |
| completed_at           | DATETIME  | Completion time                     |

### Table: `compute_assignments`

| Field           | Type      | Description                        |
| --------------- | --------- | ---------------------------------- |
| assignment_id   | TEXT / PK | Unique assignment                  |
| session_id      | TEXT / FK | Related session                    |
| data_owner_id   | TEXT / FK | Institution contributing data      |
| compute_peer_id | TEXT / FK | Institution executing Backbone     |
| round_number    | INTEGER   | Training round                     |
| status          | TEXT      | Pending, active, completed, failed |
| assigned_at     | DATETIME  | Assignment time                    |

### Table: `model_checkpoints`

| Field         | Type      | Description              |
| ------------- | --------- | ------------------------ |
| checkpoint_id | TEXT / PK | Unique checkpoint        |
| session_id    | TEXT / FK | Related session          |
| version       | INTEGER   | Backbone version         |
| storage_path  | TEXT      | Checkpoint file location |
| created_by    | TEXT      | Compute peer             |
| created_at    | DATETIME  | Creation time            |

### Table: `training_events`

| Field          | Type         | Description                                                |
| -------------- | ------------ | ---------------------------------------------------------- |
| event_id       | INTEGER / PK | Event identifier                                           |
| session_id     | TEXT / FK    | Related session                                            |
| institution_id | TEXT / FK    | Relevant institution                                       |
| event_type     | TEXT         | Registration, assignment, training, checkpoint, completion |
| message        | TEXT         | Event description                                          |
| created_at     | DATETIME     | Event timestamp                                            |

### Table: `noise_configurations`

| Field       | Type      | Description              |
| ----------- | --------- | ------------------------ |
| config_id   | TEXT / PK | Configuration identifier |
| session_id  | TEXT / FK | Related session          |
| mechanism   | TEXT      | Noise mechanism          |
| noise_scale | REAL      | Configured noise level   |
| created_at  | DATETIME  | Configuration timestamp  |

The schema is intentionally small. No separate blockchain, patient-data or attack-history tables are required.

## 3.3 Proposed repository structure

```text
medweave/
│
├── README.md
├── GLOBAL_RULES.md
├── SPEC.md
├── INTERFACES.md
│
├── plans/
│   ├── member_1/
│   │   └── PLAN.md
│   ├── member_2/
│   │   └── PLAN.md
│   ├── member_3/
│   │   └── PLAN.md
│   └── member_4/
│       └── PLAN.md
│
├── coordinator/
│   ├── flower_app/
│   ├── scheduling/
│   ├── checkpoint_manager/
│   └── relay/
│
├── split_learning/
│   ├── head.py
│   ├── backbone.py
│   ├── tail.py
│   ├── forward.py
│   └── backward.py
│
├── institutional_client/
│   ├── client.py
│   ├── local_data.py
│   └── training.py
│
├── privacy/
│   └── activation_noise.py
│
├── dashboard/
│   ├── backend/
│   └── frontend/
│
├── database/
│   ├── models.py
│   └── migrations/
│
├── tests/
│   ├── unit/
│   ├── contract/
│   └── integration/
│
├── docker/
│   └── docker-compose.yml
│
└── docs/
    ├── architecture.md
    └── setup.md
```

This is a proposed starting structure, not a requirement to create every directory in the first MVP.

## 3.4 Four-member ownership

| Member | Primary ownership      | Responsibilities                                                         |
| ------ | ---------------------- | ------------------------------------------------------------------------ |
| 1      | Coordinator and Flower | Client orchestration, scheduling, peer assignments, Backbone checkpoints |
| 2      | Split-learning engine  | Head, Backbone, Tail, forward/backward passes and gradients              |
| 3      | Dataset and privacy    | Dataset loading, IID partitioning, activation-noise implementation       |
| 4      | Dashboard and backend  | Status APIs, training-event display and later visualization              |

For the strict MVP, Member 4 can assist with integration and simulation infrastructure before building the dashboard.

Each member owns their module and must understand its implementation. All four members must understand the overall architecture at a high level.

## 3.5 Global engineering contract

The following files govern AI-assisted development:

| File              | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| `GLOBAL_RULES.md` | Rules that all members and coding agents must follow  |
| `SPEC.md`         | Master product and technical requirements             |
| `INTERFACES.md`   | Shared API contracts, schemas and function signatures |
| `PLAN.md`         | Bounded task instructions for each individual agent   |

### Development rules

1. Every member works on an independent feature branch.
2. Each AI agent receives the global rules, specification, interface contracts and its own PLAN.md.
3. Agents must implement within their assigned module.
4. Agents must not independently change shared architectural decisions.
5. New dependencies must be documented and checked for compatibility, security and licensing.
6. Every module must include tests and documentation.
7. Each pull request requires review by at least one other member.
8. CI must run formatting, linting, tests and relevant integration checks.
9. The four branches undergo a final automated integration review before the planned joint merge.
10. The last working `main` commit must be tagged before the final merge.
11. If the integrated build fails, the team must correct the problem and rerun the complete test suite.

Shared specification changes require all four members to review and reach consensus.

The aim is to prevent disconnected AI-generated modules, incompatible interfaces and code that the team cannot explain.

---

# 4. UI/UX Design System and Visual Direction

## 4.1 Design objective

The dashboard should communicate the collaborative training process visually.

It is an observability and demonstration interface, not a conversational assistant or a complete hospital-management application.

The dashboard is not part of the strict first MVP.

## 4.2 Visual direction

**Proposed direction: Professional research-infrastructure dashboard.**

The interface should feel like a technical collaborative-computing control panel.

Design characteristics:

- Clean, restrained layout.
- Strong information hierarchy.
- Minimal decorative elements.
- Clear training-state indicators.
- Visual separation of institutional clients and compute peers.
- Technical visualizations rather than marketing graphics.

Suggested palette:

| Purpose               | Suggested treatment        |
| --------------------- | -------------------------- |
| Main background       | Dark navy / charcoal       |
| Primary surfaces      | Slightly lighter dark grey |
| Primary accent        | Teal or cyan               |
| Successful operations | Green                      |
| Errors and failures   | Red                        |
| Warnings              | Amber                      |
| Secondary text        | Muted grey                 |

These are proposed design choices, not previously approved brand guidelines.

## 4.3 Main dashboard layout

### A. Overview

Display:

- MedWeave project name.
- Current session identifier.
- Training status.
- Current collaborative round.
- Number of connected institutions.
- Current data-owning institution.
- Current compute peer.

### B. Institution network

Display four institution cards.

Each card shows:

- Institution identifier.
- Connection status.
- Current role: data owner, compute peer or idle.
- Training progress.
- Latest contribution status.

### C. Training activity

A timeline showing:

1. Institution selected.
2. Compute peer assigned.
3. Activations generated.
4. Noise applied.
5. Backbone computation performed.
6. Backbone checkpoint updated.
7. Institution contribution completed.

### D. Privacy visualization

Later-phase interface:

- Original medical image.
- Activation visualization before noise.
- Activation visualization after noise.
- Relative activation change percentage.

The original image remains local to the institution. The dashboard must not expose raw institutional images through the coordinator.

### E. Model status

Display:

- Current Backbone version.
- Latest checkpoint timestamp.
- Number of completed contributions.
- Session completion state.

## 4.4 Initial UI scope

The strict MVP does not require a polished dashboard.

The first working interface can be limited to service logs, training status and basic terminal output.

The full web dashboard is a later milestone.

---

# 5. Integrations and Infrastructure

## 5.1 Development environment

The project must support mixed operating systems.

Docker and Docker Compose provide consistent software environments.

The initial development environment uses one laptop to simulate four institutions.

The eventual target is four physical laptops on the same Wi-Fi/LAN.

## 5.2 Docker Compose topology

Proposed initial topology:

```text
                    Docker Compose
                           |
              +------------+------------+
              |                         |
        Coordinator                Institution 1
        Flower service             Client container
        Scheduler
        Checkpoint store
        Relay
              |
       +------+------+------+
       |      |      |      |
   Institution 2  Institution 3  Institution 4
      Client        Client        Client
```

The four clients should have independent local-data volumes and separate runtime state.

The coordinator stores metadata and Backbone checkpoints, not raw institutional datasets.

## 5.3 Technology stack

| Layer                       | Technology                              | Status                                     |
| --------------------------- | --------------------------------------- | ------------------------------------------ |
| Language                    | Python                                  | Selected                                   |
| Deep learning               | PyTorch                                 | Selected                                   |
| Collaborative orchestration | Flower                                  | Selected                                   |
| Split-learning engine       | Custom PyTorch implementation           | Selected                                   |
| Communication               | Custom HTTP APIs + Flower communication | Agreed direction; exact boundary to freeze |
| API backend                 | FastAPI                                 | Proposed                                   |
| Frontend                    | React                                   | Proposed                                   |
| Metadata database           | SQLite                                  | Proposed                                   |
| Containers                  | Docker                                  | Selected                                   |
| Local orchestration         | Docker Compose                          | Selected                                   |
| Version control             | Git + GitHub                            | Selected                                   |
| CI                          | GitHub Actions                          | Proposed                                   |
| Dashboard                   | Web-based                               | Selected for later phase                   |

## 5.4 Network and security

- All four physical laptops should initially operate on the same local network.
- TLS must protect network communication.
- The coordinator is a trusted component.
- Compute peers are honest-but-curious.
- Raw images and labels stay on institutional clients.
- Activations are perturbed before transmission.
- The coordinator relays activations but does not store or analyse them.
- Activation payloads should not be written into persistent logs.

## 5.5 Deployment progression

**Stage 1: Single-laptop simulation**

Run four independent institution clients and the coordinator using Docker Compose.

**Stage 2: Multi-laptop deployment**

Move each institutional client to a separate physical laptop. Host the coordinator VM on one of the participating laptops.

**Stage 3: Hybrid fallback**

If hardware or network problems prevent the four-laptop demonstration, use a combination of physical laptops, VMs and containers.

The four logical institutions remain the requirement.

## 5.6 Operational fallback

If an institution disconnects:

- Mark it offline.
- Allow remaining institutions to continue.
- When it reconnects, resume its pending contribution using the latest shared Backbone checkpoint.

This is a basic recovery mechanism, not a full distributed fault-tolerance system.

---

# 6. Phased Implementation Roadmap

The project will use three implementation phases.

## Phase 1 — Working Collaborative Training MVP

<badge color="success">Highest priority</badge>

**Target:** 1–2 weeks.

**Objective:** Make the core framework work end to end on one laptop.

### Deliverables

1. Coordinator service.
2. Four independent institutional clients.
3. Flower client registration and orchestration.
4. U-shaped split-learning implementation.
5. Shared Backbone checkpoint management.
6. Sequential institutional training.
7. Rotating external compute-peer assignments.
8. Activation-noise injection.
9. One complete collaborative training run.
10. Coordinator retains the final shared Backbone.

### Acceptance criteria

- All four simulated institutions register.
- The coordinator assigns valid external compute peers.
- Each institution retains its local dataset and labels.
- Head, Backbone and Tail execute the agreed forward/backward workflow.
- The shared Backbone is updated and transferred correctly.
- Each institution completes its contribution.
- The final Backbone checkpoint is retained by the coordinator.

**Phase 1 is complete when all four simulated institutions successfully finish their sequential training contributions and the coordinator retains the final shared Backbone.**

A polished dashboard and inference demo are not required for this milestone.

## Phase 2 — Multi-Laptop Demonstration and Usability

**Objective:** Move the working framework from local simulation to a reliable demonstration environment.

### Deliverables

1. Deploy four institutional clients across four physical laptops.
2. Run the coordinator in a lightweight Linux VM.
3. Configure same-network communication and TLS.
4. Validate peer assignments and Backbone transfers across machines.
5. Add basic reconnection handling.
6. Build the dashboard backend and frontend.
7. Display institution status, training activity and model versions.
8. Add activation visualization before and after noise.
9. Demonstrate a basic inference workflow.

### Acceptance criteria

- Four physical laptops participate in collaborative training.
- Training completes without manual manipulation of internal state.
- Dashboard reflects the actual training workflow.
- The final shared model can perform basic inference.

## Phase 3 — Evaluation, Refinement and Research Documentation

**Objective:** Improve the evidence supporting the research contribution.

### Deliverables

1. Select and finalize the primary classification dataset and task.
2. Report classification accuracy on a held-out test dataset.
3. Visualize activation perturbation and its relative-change metric.
4. Improve documentation and reproducibility.
5. Evaluate additional training rounds.
6. Prepare the architecture, methodology and results for the research paper.

### Future research options

These are not initial implementation commitments:

- Formal differential privacy with clipping and privacy accounting.
- Reconstruction or inference attack experiments.
- Non-IID dataset partitioning.
- Centralized-training comparisons.
- More sophisticated peer fault tolerance.
- Blockchain governance and reputation mechanisms.
- Wider deployment across institutions.

---

# 7. Project Risks and Constraints

| Risk                                                | Impact | Mitigation                                              |
| --------------------------------------------------- | ------ | ------------------------------------------------------- |
| Split-learning integration complexity               | High   | Build a minimal end-to-end training path first          |
| Incompatible AI-generated modules                   | High   | Freeze interface contracts and use CI                   |
| Flower customization difficulties                   | High   | Validate a minimal custom workflow early                |
| Gradient-transfer errors                            | High   | Write focused forward/backward tests                    |
| Inconsistent environments                           | Medium | Docker Compose                                          |
| Laptop resource limitations                         | Medium | Use a small model and dataset                           |
| Network communication failures                      | Medium | Start with local simulation                             |
| Activation visualization mistaken for privacy proof | High   | Explicitly distinguish perturbation from proven privacy |
| Scope expansion                                     | High   | Protect the strict MVP and defer optional features      |

---

# 8. Deferred Decisions and Open Items

The following are deliberately not frozen yet.

| Item                                           | Reason                                        |
| ---------------------------------------------- | --------------------------------------------- |
| Exact dataset                                  | Requires dataset suitability review           |
| Classification task and labels                 | Depends on dataset selection                  |
| Model architecture                             | Must follow dataset/task selection            |
| Exact noise mechanism and scale                | Requires implementation-level experimentation |
| Formal DP accounting                           | Outside initial scope                         |
| Flower strategy and custom application details | Requires prototype validation                 |
| Exact HTTP API schemas                         | To be frozen in `INTERFACES.md`               |
| Dashboard component library                    | Not needed for first MVP                      |
| TLS certificate management                     | Finalize for multi-laptop deployment          |
| Final database implementation                  | SQLite proposed; schema may evolve            |
| Detailed training hyperparameters              | Determine after selecting the task            |

---

# 9. Final Definition of MedWeave

MedWeave is an academic proof-of-concept privacy-oriented collaborative AI framework.

Its primary contribution is a functioning system that enables four simulated institutions to collaboratively train a shared model using U-shaped split learning and rotating peer computation, while retaining their raw training datasets locally.

The framework uses a lightweight coordinator for orchestration and shared Backbone management, Flower for collaborative coordination, PyTorch for model computation, and noise perturbation for intermediate activations.

The initial goal is not to prove clinical effectiveness or formal privacy guarantees. It is to build and demonstrate a complete collaborative training workflow.

**Primary project success criterion:** Four independent institutional clients successfully complete collaborative training, using computation performed by other participating peers, and the coordinator retains the final shared Backbone.

**Primary SDG alignment:** SDG 9 — Industry, Innovation and Infrastructure.

**Guiding engineering principle: Build the working system first. Simplify everything else if necessary.**
