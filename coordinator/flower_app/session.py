"""The coordinator's session loop: register institutions, then run each turn.

Written against a minimal `Grid` interface so it can be tested without
starting Flower processes. `server_app.py` passes Flower's real `Grid`.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from flwr.app import ArrayRecord, Message, RecordDict

from coordinator.checkpoint_manager.manager import CheckpointManager, CheckpointRecord
from coordinator.flower_app import protocol
from coordinator.flower_app.observer import SessionObserver
from coordinator.flower_app.protocol import TurnInfo
from coordinator.relay.relay import Relay
from coordinator.scheduling.schedule import Turn
from split_learning.compute_peer import BackboneUpdate

Log = Callable[[str], None]


class Grid(Protocol):
    def get_node_ids(self) -> Iterable[int]: ...

    def send_and_receive(self, messages: Iterable[Message], *, timeout: float | None = None) -> Iterable[Message]: ...


class SessionError(Exception):
    """Registration or a training turn failed; the Backbone was not updated."""


@dataclass(frozen=True)
class TurnResult:
    turn: Turn
    checkpoint: CheckpointRecord
    loss: float
    accuracy: float
    num_batches: int
    seconds: float


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
    observer: SessionObserver | None = None,
) -> list[TurnResult]:
    """Run every turn in order; each accepted turn adds one Backbone version.

    The observer must already have been told `session_started`.
    """
    observer = observer or SessionObserver()
    results = []
    try:
        for turn in schedule:
            connected = set(grid.get_node_ids())
            observer.connection_checked({member: node in connected for member, node in nodes.items()})
            result = run_turn(grid, nodes, turn, manager, relay, relay_url, num_batches, turn_timeout, log, observer)
            results.append(result)
    except Exception as error:
        observer.session_failed(str(error))
        raise
    log(f"session complete: coordinator holds Backbone v{manager.latest_version}")
    observer.session_finished(manager.latest_version)
    return results


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
    )
    log(
        f"  done in {result.seconds:.1f}s: {result.num_batches} batches, loss {result.loss:.3f}, "
        f"accuracy {result.accuracy:.2f} -> Backbone v{checkpoint.version}"
    )
    return result
