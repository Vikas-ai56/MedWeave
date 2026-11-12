"""Sequential training schedule.

Each round, every institution takes one turn as the data owner. Its compute
peer is the next institution in the given order, wrapping around at the end,
so no institution ever runs the Backbone on its own activations. With four
institutions a round is 1->2, 2->3, 3->4, 4->1 (SPEC §2.4).
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace


class SchedulingError(Exception):
    """A turn cannot be scheduled, e.g. no compute peer is available."""


@dataclass(frozen=True)
class Turn:
    """One institution's training turn and the peer that runs its Backbone.

    `round_number` starts at 1.
    """

    round_number: int
    data_owner_id: str
    compute_peer_id: str

    def __post_init__(self) -> None:
        if self.data_owner_id == self.compute_peer_id:
            raise ValueError(f"{self.data_owner_id!r} cannot be its own compute peer")


def build_schedule(institution_ids: Sequence[str], num_rounds: int = 1) -> list[Turn]:
    """Return every turn of the session, in the order they should run."""
    ids = _check_ids(institution_ids)
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be at least 1, got {num_rounds}")
    return [
        Turn(round_number, owner, ids[(index + 1) % len(ids)])
        for round_number in range(1, num_rounds + 1)
        for index, owner in enumerate(ids)
    ]


def reassign_peer(turn: Turn, institution_ids: Sequence[str], online_ids: Collection[str]) -> Turn:
    """Replace an offline compute peer with the next online institution.

    The search starts after the current peer in `institution_ids` order and
    skips the data owner, so the result is deterministic. The turn is returned
    unchanged if its peer is online. Whether the data owner itself is online is
    not checked here.
    """
    ids = _check_ids(institution_ids)
    for member in (turn.data_owner_id, turn.compute_peer_id):
        if member not in ids:
            raise ValueError(f"{member!r} is not a registered institution")
    if turn.compute_peer_id in online_ids:
        return turn

    start = ids.index(turn.compute_peer_id)
    for offset in range(1, len(ids)):
        candidate = ids[(start + offset) % len(ids)]
        if candidate != turn.data_owner_id and candidate in online_ids:
            return replace(turn, compute_peer_id=candidate)
    raise SchedulingError(f"no online compute peer available for {turn.data_owner_id!r}")


def _check_ids(institution_ids: Sequence[str]) -> list[str]:
    ids = list(institution_ids)
    if len(ids) < 2:
        raise ValueError(f"need at least 2 institutions, got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"institution ids must be unique, got {ids}")
    return ids
