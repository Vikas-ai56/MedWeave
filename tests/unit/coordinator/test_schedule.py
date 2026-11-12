import pytest

from coordinator.scheduling.schedule import SchedulingError, Turn, build_schedule, reassign_peer

IDS = ["inst-1", "inst-2", "inst-3", "inst-4"]


def test_four_institutions_rotate_to_the_next_peer():
    assert build_schedule(IDS) == [
        Turn(1, "inst-1", "inst-2"),
        Turn(1, "inst-2", "inst-3"),
        Turn(1, "inst-3", "inst-4"),
        Turn(1, "inst-4", "inst-1"),
    ]


def test_two_institutions_compute_for_each_other():
    assert build_schedule(["a", "b"]) == [Turn(1, "a", "b"), Turn(1, "b", "a")]


@pytest.mark.parametrize("count", [2, 3, 4, 7])
@pytest.mark.parametrize("num_rounds", [1, 3])
def test_no_institution_is_its_own_peer(count, num_rounds):
    ids = [f"inst-{i}" for i in range(count)]
    schedule = build_schedule(ids, num_rounds)
    assert all(turn.data_owner_id != turn.compute_peer_id for turn in schedule)


def test_each_institution_owns_one_turn_per_round():
    schedule = build_schedule(IDS, num_rounds=3)
    assert len(schedule) == 12
    for round_number in (1, 2, 3):
        owners = [turn.data_owner_id for turn in schedule if turn.round_number == round_number]
        assert sorted(owners) == sorted(IDS)
    assert [turn.round_number for turn in schedule] == sorted(turn.round_number for turn in schedule)


def test_same_input_gives_same_schedule():
    assert build_schedule(IDS, 2) == build_schedule(tuple(IDS), 2)


@pytest.mark.parametrize(
    ("ids", "num_rounds"),
    [
        ([], 1),
        (["only"], 1),
        (["a", "a", "b"], 1),
        (IDS, 0),
        (IDS, -1),
    ],
)
def test_invalid_input_raises(ids, num_rounds):
    with pytest.raises(ValueError):
        build_schedule(ids, num_rounds)


def test_turn_rejects_self_as_peer():
    with pytest.raises(ValueError):
        Turn(1, "inst-1", "inst-1")


def test_reassign_keeps_an_online_peer():
    turn = Turn(1, "inst-1", "inst-2")
    assert reassign_peer(turn, IDS, set(IDS)) is turn


def test_reassign_picks_the_next_online_peer():
    turn = Turn(2, "inst-1", "inst-2")
    assert reassign_peer(turn, IDS, {"inst-1", "inst-3", "inst-4"}) == Turn(2, "inst-1", "inst-3")


def test_reassign_skips_several_offline_peers():
    turn = Turn(1, "inst-1", "inst-2")
    assert reassign_peer(turn, IDS, {"inst-1", "inst-4"}) == Turn(1, "inst-1", "inst-4")


def test_reassign_wraps_around_and_never_picks_the_data_owner():
    turn = Turn(1, "inst-3", "inst-4")
    assert reassign_peer(turn, IDS, {"inst-2", "inst-3"}) == Turn(1, "inst-3", "inst-2")


def test_reassign_raises_when_no_peer_is_online():
    turn = Turn(1, "inst-1", "inst-2")
    with pytest.raises(SchedulingError):
        reassign_peer(turn, IDS, {"inst-1"})


def test_reassign_rejects_unknown_institutions():
    with pytest.raises(ValueError):
        reassign_peer(Turn(1, "inst-9", "inst-2"), IDS, set(IDS))
    with pytest.raises(ValueError):
        reassign_peer(Turn(1, "inst-1", "inst-9"), IDS, set(IDS))
