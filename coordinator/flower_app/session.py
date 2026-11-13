"""The coordinator's session loop: register institutions, run each turn, evaluate after each round.

Written against a minimal `Grid` interface so it can be tested without
starting Flower processes. `server_app.py` passes Flower's real `Grid`.

After the last turn of every round (when evaluation is enabled) every institution
receives the latest Backbone and scores its own Head and Tail on a public held-out
split locally; only loss, accuracy and sample count come back.

Every accepted turn is tracked at step = its new Backbone version (train loss and
accuracy, activation change, seconds) and every evaluation at step = round number
(per-institution and mean validation accuracy), through the `tracking.Tracker` given.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import groupby
from typing import Protocol

from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict

from coordinator.checkpoint_manager.manager import CheckpointManager, CheckpointRecord
from coordinator.flower_app import protocol
from coordinator.flower_app.observer import SessionObserver
from coordinator.flower_app.protocol import TurnInfo
from coordinator.relay.relay import Relay
from coordinator.scheduling.schedule import Turn
from split_learning.compute_peer import BackboneUpdate
from tracking import NullTracker, Tracker

Log = Callable[[str], None]


class Grid(Protocol):
    def get_node_ids(self) -> Iterable[int]: ...

    def send_and_receive(self, messages: Iterable[Message], *, timeout: float | None = None) -> Iterable[Message]: ...


class SessionError(Exception):
    """Registration, a training turn or an evaluation failed; a failed turn leaves the Backbone unchanged."""


@dataclass(frozen=True)
class TurnResult:
    turn: Turn
    checkpoint: CheckpointRecord
    loss: float
    accuracy: float
    num_batches: int
    seconds: float
    noise_std: float
    activation_change_pct: float


@dataclass(frozen=True)
class EvaluationSettings:
    split: str
    batch_size: int
    timeout: float


@dataclass(frozen=True)
class RoundEvaluation:
    round_number: int
    backbone_version: int
    split: str
    accuracy: dict[str, float]
    loss: dict[str, float]
    num_samples: int

    @property
    def mean_accuracy(self) -> float:
        return sum(self.accuracy.values()) / len(self.accuracy)


@dataclass(frozen=True)
class SessionResult:
    turns: list[TurnResult]
    evaluations: list[RoundEvaluation]


def register_institutions(
    grid: Grid,
    expected: int,
    timeout: float,
    poll_interval: float = 2.0,
    log: Log = print,
    observer: SessionObserver | None = None,
) -> dict[str, int]:
    """Wait for `expected` SuperNodes and return {institution_id: Flower node id}."""
    deadline = time.monotonic() + timeout
    while len(node_ids := list(grid.get_node_ids())) < expected:
        if time.monotonic() > deadline:
            raise SessionError(f"only {len(node_ids)} of {expected} institutions connected within {timeout}s")
        log(f"waiting for institutions: {len(node_ids)}/{expected} connected")
        time.sleep(poll_interval)

    queries = [Message(RecordDict(), dst_node_id=node, message_type=protocol.REGISTER) for node in node_ids]
    nodes: dict[str, int] = {}
    for reply in grid.send_and_receive(queries, timeout=timeout):
        if reply.has_error():
            raise SessionError(f"node {reply.metadata.src_node_id} failed to register: {reply.error}")
        institution_id = str(reply.content[protocol.INFO][protocol.INSTITUTION_ID])
        if institution_id in nodes:
            raise SessionError(f"two nodes registered as {institution_id!r}")
        nodes[institution_id] = reply.metadata.src_node_id
    if len(nodes) != len(node_ids):
        raise SessionError(f"{len(node_ids) - len(nodes)} node(s) did not answer registration")
    log(f"registered institutions: {', '.join(sorted(nodes))}")
    (observer or SessionObserver()).institutions_registered(nodes)
    return nodes


def run_session(
    grid: Grid,
    nodes: dict[str, int],
    schedule: Iterable[Turn],
    manager: CheckpointManager,
    relay: Relay,
    relay_url: str,
    num_batches: int,
    turn_timeout: float,
    log: Log = print,
    evaluation: EvaluationSettings | None = None,
    tracker: Tracker | None = None,
    observer: SessionObserver | None = None,
) -> SessionResult:
    """Run every turn in order; each accepted turn adds one Backbone version.

    The observer must already have been told `session_started`.
    """
    tracker = tracker or NullTracker()
    observer = observer or SessionObserver()
    turns: list[TurnResult] = []
    evaluations: list[RoundEvaluation] = []
    try:
        for round_number, round_turns in groupby(schedule, key=lambda turn: turn.round_number):
            for turn in round_turns:
                connected = set(grid.get_node_ids())
                observer.connection_checked({member: node in connected for member, node in nodes.items()})
                result = run_turn(
                    grid, nodes, turn, manager, relay, relay_url, num_batches, turn_timeout, log, observer
                )
                tracker.log_metrics(turn_metrics(result), step=result.checkpoint.version)
                turns.append(result)
            if evaluation is not None:
                round_evaluation = evaluate_institutions(grid, nodes, manager, round_number, evaluation, log)
                tracker.log_metrics(evaluation_metrics(round_evaluation), step=round_number)
                evaluations.append(round_evaluation)
    except Exception as error:
        observer.session_failed(str(error))
        raise
    log(f"session complete: coordinator holds Backbone v{manager.latest_version}")
    observer.session_finished(manager.latest_version)
    return SessionResult(turns, evaluations)

def turn_metrics(result: TurnResult) -> dict[str, float]:
    return {
        "train_loss": result.loss,
        "train_accuracy": result.accuracy,
        "activation_change_pct": result.activation_change_pct,
        "turn_seconds": result.seconds,
        f"train_accuracy/{result.turn.data_owner_id}": result.accuracy,
    }


def evaluation_metrics(evaluation: RoundEvaluation) -> dict[str, float]:
    prefix = f"{evaluation.split}_accuracy"
    return {
        **{f"{prefix}/{name}": accuracy for name, accuracy in evaluation.accuracy.items()},
        **{f"{evaluation.split}_loss/{name}": loss for name, loss in evaluation.loss.items()},
        f"{prefix}_mean": evaluation.mean_accuracy,
    }


def evaluate_institutions(
    grid: Grid,
    nodes: dict[str, int],
    manager: CheckpointManager,
    round_number: int,
    settings: EvaluationSettings,
    log: Log = print,
) -> RoundEvaluation:
    state, version = manager.latest()
    request = ConfigRecord({"backbone_version": version, "split": settings.split, "batch_size": settings.batch_size})
    messages = [
        Message(
            RecordDict({protocol.EVALUATION: request, protocol.BACKBONE: ArrayRecord(torch_state_dict=state)}),
            dst_node_id=node,
            message_type=protocol.EVALUATE_HELD_OUT,
            ttl=settings.timeout,
        )
        for node in nodes.values()
    ]
    replies = {reply.metadata.src_node_id: reply for reply in grid.send_and_receive(messages, timeout=settings.timeout)}
    problems = []
    for name, node in sorted(nodes.items()):
        if node not in replies:
            problems.append(f"{name} did not reply within {settings.timeout}s")
        elif replies[node].has_error():
            problems.append(f"{name} failed: {replies[node].error.reason}")
    if problems:
        raise SessionError(f"round {round_number} evaluation failed: " + "; ".join(problems))

    metrics = {name: replies[node].content[protocol.METRICS] for name, node in sorted(nodes.items())}
    result = RoundEvaluation(
        round_number=round_number,
        backbone_version=int(version),
        split=settings.split,
        accuracy={name: float(record["accuracy"]) for name, record in metrics.items()},
        loss={name: float(record["loss"]) for name, record in metrics.items()},
        num_samples=int(next(iter(metrics.values()))["num_samples"]),
    )
    scores = ", ".join(f"{name} {accuracy:.3f}" for name, accuracy in result.accuracy.items())
    log(
        f"round {round_number} {settings.split} accuracy with Backbone v{version} "
        f"({result.num_samples} samples): {scores}; mean {result.mean_accuracy:.3f}"
    )
    return result


def run_turn(
    grid: Grid,
    nodes: dict[str, int],
    turn: Turn,
    manager: CheckpointManager,
    relay: Relay,
    relay_url: str,
    num_batches: int,
    turn_timeout: float,
    log: Log = print,
    observer: SessionObserver | None = None,
) -> TurnResult:
    """Send the Backbone to the peer and start the owner, then commit the peer's update."""
    for member in (turn.data_owner_id, turn.compute_peer_id):
        if member not in nodes:
            raise SessionError(f"{member!r} is not registered")
    observer = observer or SessionObserver()
    state, version = manager.latest()
    observer.turn_started(turn, version, num_batches)
    try:
        result = _exchange(grid, nodes, turn, manager, relay, relay_url, num_batches, turn_timeout, log, state, version)
    except Exception as error:
        observer.turn_failed(turn, str(error))
        raise
    observer.checkpoint_saved(result.checkpoint)
    observer.turn_finished(turn, result.loss, result.accuracy, result.num_batches)
    return result


def _exchange(
    grid: Grid,
    nodes: dict[str, int],
    turn: Turn,
    manager: CheckpointManager,
    relay: Relay,
    relay_url: str,
    num_batches: int,
    turn_timeout: float,
    log: Log,
    state: dict,
    version: str,
) -> TurnResult:
    info = TurnInfo(
        turn.round_number, turn.data_owner_id, turn.compute_peer_id, version, relay_url, num_batches, turn_timeout
    ).to_record()
    peer_node, owner_node = nodes[turn.compute_peer_id], nodes[turn.data_owner_id]
    messages = [
        Message(
            RecordDict({protocol.TURN: info, protocol.BACKBONE: ArrayRecord(torch_state_dict=state)}),
            dst_node_id=peer_node,
            message_type=protocol.TRAIN_AS_PEER,
            ttl=turn_timeout,
        ),
        Message(
            RecordDict({protocol.TURN: info}),
            dst_node_id=owner_node,
            message_type=protocol.TRAIN_AS_OWNER,
            ttl=turn_timeout,
        ),
    ]

    log(
        f"round {turn.round_number}: {turn.data_owner_id} trains, "
        f"{turn.compute_peer_id} computes, Backbone v{version}"
    )
    started = time.monotonic()
    relay.open_turn(turn.data_owner_id, turn.compute_peer_id, version)
    try:
        replies = {reply.metadata.src_node_id: reply for reply in grid.send_and_receive(messages, timeout=turn_timeout)}
    finally:
        relay.close_turn()

    problems = []  # report both sides: one failure often causes the other (e.g. a timeout)
    for node, name in ((peer_node, turn.compute_peer_id), (owner_node, turn.data_owner_id)):
        if node not in replies:
            problems.append(f"{name} did not reply within {turn_timeout}s")
        elif replies[node].has_error():
            problems.append(f"{name} failed: {replies[node].error.reason}")
    if problems:
        raise SessionError(f"turn {turn} not committed: " + "; ".join(problems))

    peer_reply = replies[peer_node].content
    update = BackboneUpdate(
        base_version=str(peer_reply[protocol.UPDATE]["base_version"]),
        state_dict=peer_reply[protocol.BACKBONE].to_torch_state_dict(),
        num_batches=int(peer_reply[protocol.UPDATE]["num_batches"]),
    )
    if update.num_batches != num_batches:
        raise SessionError(f"{turn.compute_peer_id} ran {update.num_batches} of {num_batches} batches")
    checkpoint = manager.commit(update, created_by=turn.compute_peer_id)

    metrics = replies[owner_node].content[protocol.METRICS]
    result = TurnResult(
        turn,
        checkpoint,
        loss=float(metrics["loss"]),
        accuracy=float(metrics["accuracy"]),
        num_batches=update.num_batches,
        seconds=time.monotonic() - started,
        noise_std=float(metrics["noise_std"]),
        activation_change_pct=float(metrics["activation_change_pct"]),
    )
    log(
        f"  done in {result.seconds:.1f}s: {result.num_batches} batches, loss {result.loss:.3f}, "
        f"accuracy {result.accuracy:.2f}, noise std {result.noise_std:g} "
        f"(activations changed {result.activation_change_pct:.1f}%) -> Backbone v{checkpoint.version}"
    )
    return result
