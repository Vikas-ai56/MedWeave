MedWeave

Privacy-Preserving Collaborative AI Framework

MedWeave is a collaborative AI framework that allows multiple institutions to train a shared machine learning model without sharing their raw datasets.

The project explores a combination of U-shaped Split Learning and collaborative training, where institutions retain their local data while using computing resources from other participating institutions.

Overview

Sharing data between institutions can be difficult due to privacy concerns, institutional policies, and limited computing resources.

MedWeave aims to address this by separating data ownership from model computation. Each institution keeps its dataset locally and participates in training through an exchange of intermediate model representations.

The initial implementation uses four simulated institutions running on a single machine. The system is designed to be extended to multiple machines connected over a local network.

How It Works

Each institution maintains its own local dataset.

A local model processes the input data and generates intermediate activations.

The activations are sent to an assigned compute peer for further processing.

The resulting gradients are returned to the institution.

Institutions take turns contributing to a shared Backbone model.

The coordinator manages training and maintains the latest shared model checkpoint.

Raw images and labels remain with their respective institutions.

Architecture

MedWeave consists of four main components:

Coordinator: Manages institutions, training schedules, compute assignments, and shared model checkpoints.

Institutional Clients: Hold local datasets and execute the institution-specific parts of the model.

Compute Peers: Perform the remote Backbone computation assigned to them.

Privacy Layer: Supports configurable noise injection into intermediate activations before transmission.

Technology Stack

Component

Technology

Programming Language

Python

Deep Learning

PyTorch

Training Orchestration

Flower

Backend API

FastAPI

Frontend

React

Database

SQLite

Containerization

Docker, Docker Compose

Version Control

Git, GitHub

Project Structure

MedWeave/
├── coordinator/ # Training coordination and model management
├── institutional_client/ # Local institutional training logic
├── split_learning/ # Head, Backbone and Tail implementation
├── privacy/ # Activation noise and related utilities
├── dashboard/ # Monitoring interface
├── tests/ # Unit and integration tests
├── docker/ # Container configuration
├── docs/ # Architecture and project documentation
├── GLOBAL_RULES.md
├── SPEC.md
├── INTERFACES.md
└── README.md

The structure may evolve as implementation progresses.

Current Development Scope

The initial milestone focuses on a working end-to-end prototype:

Four simulated institutions.

Local datasets with separate data loaders.

U-shaped Split Learning implementation.

Coordinator-managed sequential training.

Rotating compute peer assignments.

Configurable activation noise.

A shared Backbone checkpoint maintained by the coordinator.

The prototype will initially run on a single machine using Docker Compose, with multi-machine execution planned as a subsequent milestone.

Getting Started

Prerequisites

Python 3.11+

Docker and Docker Compose

Git

Clone the Repository

git clone <repository-url>
cd MedWeave

Run the Project

Setup and execution instructions will be added as the initial implementation is integrated.

Team

Developed as an academic project by a team of four students at RV College of Engineering, Bengaluru.

MedWeave is an academic proof-of-concept. It is not intended for clinical use and does not currently provide a formally verified differential privacy guarantee.
