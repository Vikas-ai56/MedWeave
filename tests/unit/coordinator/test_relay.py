import threading

import pytest
import torch

from coordinator.relay.relay import (
    Envelope,
    Kind,
    Relay,
    RelayError,
    RelayTimeout,
    decode_tensor,
    encode_tensor,
)
from coordinator.relay.transport import RelayClient, RelayServer

OWNER, PEER, VERSION = "inst-1", "inst-2", "3"


def envelope(kind=Kind.ACTIVATIONS, institution_id=OWNER, version=VERSION, body=b"x"):
    return Envelope(kind, institution_id, "inst-1-v3-0", version, body)


@pytest.fixture
def relay() -> Relay:
    relay = Relay()
    relay.open_turn(OWNER, PEER, VERSION)
    return relay


@pytest.fixture
def server(relay):
    with RelayServer(relay, port=0) as server:
        yield server


# --- relay rules ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "recipient"),
    [
        (Kind.ACTIVATIONS, PEER),
        (Kind.BACKBONE_OUTPUT, OWNER),
        (Kind.GRADIENT, PEER),
        (Kind.INPUT_GRADIENT, OWNER),
        (Kind.DONE, PEER),
    ],
)
def test_each_kind_reaches_its_recipient(relay, kind, recipient):
    sent = envelope(kind)
    relay.send(recipient, sent)
    assert relay.receive(recipient, timeout=1) == sent


@pytest.mark.parametrize(
    ("kind", "wrong_recipient"),
    [(Kind.ACTIVATIONS, OWNER), (Kind.BACKBONE_OUTPUT, PEER), (Kind.DONE, OWNER)],
)
def test_message_to_the_wrong_side_is_rejected(relay, kind, wrong_recipient):
    with pytest.raises(RelayError):
        relay.send(wrong_recipient, envelope(kind))


def test_message_for_another_data_owner_is_rejected(relay):
    with pytest.raises(RelayError):
        relay.send(PEER, envelope(institution_id="inst-3"))


def test_message_for_another_backbone_version_is_rejected(relay):
    with pytest.raises(RelayError):
        relay.send(PEER, envelope(version="2"))


def test_outsider_cannot_receive(relay):
    with pytest.raises(RelayError):
        relay.receive("inst-3", timeout=0.1)


def test_nothing_is_relayed_without_an_open_turn():
    relay = Relay()
    with pytest.raises(RelayError):
        relay.send(PEER, envelope())
    with pytest.raises(RelayError):
        relay.receive(PEER, timeout=0.1)


def test_receive_times_out(relay):
    with pytest.raises(RelayTimeout):
        relay.receive(PEER, timeout=0.05)


def test_closing_a_turn_drops_undelivered_messages(relay):
    relay.send(PEER, envelope())
    relay.close_turn()
    assert relay.active_turn is None
    relay.open_turn(OWNER, PEER, VERSION)
    with pytest.raises(RelayTimeout):
        relay.receive(PEER, timeout=0.05)


def test_only_one_turn_at_a_time(relay):
    with pytest.raises(RelayError):
        relay.open_turn("inst-3", "inst-4", VERSION)


def test_institution_cannot_relay_to_itself():
    with pytest.raises(ValueError):
        Relay().open_turn(OWNER, OWNER, VERSION)


def test_tensor_round_trip():
    tensor = torch.randn(2, 3, 4)
    assert torch.equal(decode_tensor(encode_tensor(tensor)), tensor)


@pytest.mark.parametrize("body", [b"garbage", b""])
def test_body_that_is_not_a_tensor_is_rejected(body):
    with pytest.raises(RelayError):
        decode_tensor(body)


# --- HTTP transport ------------------------------------------------------------


def test_http_round_trip_keeps_every_field(server):
    client = RelayClient(server.url)
    sent = envelope(body=encode_tensor(torch.ones(2, 2)))
    client.send(PEER, sent)
    received = client.receive(PEER, timeout=5)
    assert received == sent
    assert torch.equal(decode_tensor(received.body), torch.ones(2, 2))


def test_http_receive_waits_for_a_late_message(server):
    client = RelayClient(server.url)
    timer = threading.Timer(0.3, client.send, args=(PEER, envelope()))
    timer.start()
    assert client.receive(PEER, timeout=5).kind is Kind.ACTIVATIONS
    timer.join()


def test_http_refused_message_raises(server):
    with pytest.raises(RelayError, match="403"):
        RelayClient(server.url).send(PEER, envelope(version="2"))


def test_http_receive_times_out(server):
    with pytest.raises(RelayTimeout):
        RelayClient(server.url).receive(PEER, timeout=0.3)


def test_unreachable_relay_raises():
    with pytest.raises(RelayError):
        RelayClient("http://127.0.0.1:9").send(PEER, envelope())


def test_http_server_writes_no_logs(server, capfd):
    client = RelayClient(server.url)
    client.send(PEER, envelope())
    client.receive(PEER, timeout=5)
    captured = capfd.readouterr()
    assert captured.out == "" and captured.err == ""
