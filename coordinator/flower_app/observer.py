"""Session observers: what the coordinator reports while a session runs.

`SessionObserver` does nothing and is the default. `DatabaseObserver` writes
everything to the metadata store (`database/`), which is what the dashboard
reads. Observers only receive metadata (ids, versions, counts, metrics): the
relay tells them a message's kind and batch id, never its contents.
"""

from coordinator.checkpoint_manager.manager import CheckpointRecord
from coordinator.relay.relay import Kind
from coordinator.scheduling.schedule import Turn
from database.store import MetadataStore


class SessionObserver:
    """No-op base class. Override the hooks you need."""

    def institutions_registered(self, nodes: dict[str, int]) -> None: ...

    def session_started(self, schedule: list[Turn], num_rounds: int) -> None: ...

    def checkpoint_saved(self, record: CheckpointRecord) -> None: ...

    def connection_checked(self, online: dict[str, bool]) -> None: ...

    def turn_started(self, turn: Turn, backbone_version: str, num_batches: int) -> None: ...

    def message_relayed(self, kind: Kind, institution_id: str, batch_id: str) -> None: ...

    def turn_finished(self, turn: Turn, loss: float, accuracy: float, num_batches: int) -> None: ...

    def turn_failed(self, turn: Turn, reason: str) -> None: ...

    def session_finished(self, backbone_version: int | None) -> None: ...

    def session_failed(self, reason: str) -> None: ...


class DatabaseObserver(SessionObserver):
    """Records the session in SQLite for the dashboard (SPEC §3.2 tables)."""

    def __init__(self, store: MetadataStore) -> None:
        self.store = store
        self.session_id: str | None = None
        self._pending_registrations: list[str] = []
        self._assignment_id: str | None = None
        self._first_activation_seen = False
        self._finished = False

    def institutions_registered(self, nodes: dict[str, int]) -> None:
        for institution_id, node_id in sorted(nodes.items()):
            self.store.register_institution(institution_id, client_address=f"flower-node:{node_id}")
            self._pending_registrations.append(institution_id)

    def session_started(self, schedule: list[Turn], num_rounds: int) -> None:
        self.session_id = self.store.start_session(num_rounds, total_turns=len(schedule))
        for institution_id in self._pending_registrations:
            self.store.add_event(self.session_id, institution_id, "registration", f"{institution_id} registered")
        self._pending_registrations.clear()

    def checkpoint_saved(self, record: CheckpointRecord) -> None:
        session_id = self._require_session()
        self.store.add_checkpoint(
            session_id, record.version, str(record.storage_path), record.created_by, record.created_at.isoformat()
        )
        source = "initialised by the coordinator" if record.version == 0 else f"computed by {record.created_by}"
        self.store.add_event(session_id, None, "checkpoint", f"Backbone v{record.version} saved ({source})")

    def connection_checked(self, online: dict[str, bool]) -> None:
        for institution_id, is_online in online.items():
            if not is_online:
                self.store.set_institution_status(institution_id, "offline", seen=False)

    def turn_started(self, turn: Turn, backbone_version: str, num_batches: int) -> None:
        session_id = self._require_session()
        owner, peer = turn.data_owner_id, turn.compute_peer_id
        self.store.set_current_turn(session_id, turn.round_number, owner, peer)
        self.store.set_institution_status(owner, "training")
        self.store.set_institution_status(peer, "training")
        self._first_activation_seen = False
        self._assignment_id = self.store.start_assignment(
            session_id, turn.round_number, owner, peer, int(backbone_version), num_batches
        )
        self.store.add_event(session_id, owner, "assignment", f"Round {turn.round_number}: {owner} selected as data owner")
        self.store.add_event(
            session_id, peer, "assignment", f"{peer} assigned as compute peer, Backbone v{backbone_version} sent"
        )

    def message_relayed(self, kind: Kind, institution_id: str, batch_id: str) -> None:
        assignment_id = self._assignment_id
        if assignment_id is None:
            return
        if kind is Kind.ACTIVATIONS and not self._first_activation_seen:
            self._first_activation_seen = True
            self.store.mark_first_activation(assignment_id)
            self.store.add_event(
                self._require_session(), institution_id, "training", f"{institution_id} started sending activations"
            )
        elif kind is Kind.INPUT_GRADIENT:
            self.store.record_batch_done(assignment_id)

    def turn_finished(self, turn: Turn, loss: float, accuracy: float, num_batches: int) -> None:
        session_id = self._require_session()
        if self._assignment_id is not None:
            self.store.complete_assignment(self._assignment_id, loss, accuracy)
        self._assignment_id = None
        for member in (turn.data_owner_id, turn.compute_peer_id):
            self.store.set_institution_status(member, "online")
        self.store.add_event(
            session_id,
            turn.data_owner_id,
            "completion",
            f"{turn.data_owner_id} finished its contribution: {num_batches} batches, "
            f"loss {loss:.3f}, accuracy {accuracy:.0%}",
        )

    def turn_failed(self, turn: Turn, reason: str) -> None:
        session_id = self._require_session()
        if self._assignment_id is not None:
            self.store.fail_assignment(self._assignment_id, reason)
        self._assignment_id = None
        for member in (turn.data_owner_id, turn.compute_peer_id):
            self.store.set_institution_status(member, "online", seen=False)
        self.store.add_event(session_id, turn.data_owner_id, "failure", f"Turn failed: {reason}")

    def session_finished(self, backbone_version: int | None) -> None:
        session_id = self._require_session()
        self._finished = True
        self.store.finish_session(session_id, "completed")
        self.store.add_event(
            session_id, None, "completion", f"Session complete: coordinator holds Backbone v{backbone_version}"
        )

    def session_failed(self, reason: str) -> None:
        if self.session_id is None or self._finished:
            return  # nothing to record against yet, or already recorded
        self._finished = True
        self.store.finish_session(self.session_id, "failed")
        self.store.add_event(self.session_id, None, "failure", f"Session stopped: {reason}")

    def _require_session(self) -> str:
        if self.session_id is None:
            raise RuntimeError("session_started must be called first")
        return self.session_id
