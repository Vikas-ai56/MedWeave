"""A fake Flower Grid that delivers messages to the real ClientApp handlers."""

from concurrent.futures import ThreadPoolExecutor

from flwr.app import Context, Error, Message, RecordDict

from coordinator.flower_app import client_app, protocol

IDS = ["inst-1", "inst-2", "inst-3", "inst-4"]
RUN_CONFIG = {"optimizer": "adam", "learning-rate": 0.001, "partition-seed": 0, "batch-size": 4, "noise-std": 0.1, "noise-seed": 0}
HANDLERS = {
    protocol.REGISTER: client_app.register,
    protocol.TRAIN_AS_PEER: client_app.compute_for_peer,
    protocol.TRAIN_AS_OWNER: client_app.train_own_data,
    protocol.EVALUATE_HELD_OUT: client_app.evaluate_locally,
}


class FakeGrid:
    """Delivers each message to the ClientApp handler of its node, concurrently."""

    def __init__(self, institution_ids, failing=(), silent=()):
        self.contexts = {
            100 + index: Context(
                run_id=1,
                node_id=100 + index,
                node_config={protocol.INSTITUTION_ID: institution_id},
                state=RecordDict(),
                run_config=dict(RUN_CONFIG),
            )
            for index, institution_id in enumerate(institution_ids)
        }
        self.failing = set(failing)  # (institution_id, message_type) pairs that raise
        self.silent = set(silent)  # institution ids whose replies get lost

    def node_of(self, institution_id):
        return next(n for n, c in self.contexts.items() if c.node_config[protocol.INSTITUTION_ID] == institution_id)

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self, messages, *, timeout=None):
        messages = list(messages)
        with ThreadPoolExecutor(max_workers=len(messages)) as pool:
            replies = list(pool.map(self._deliver, messages))
        return [reply for reply in replies if reply is not None]

    def _deliver(self, msg):
        context = self.contexts[msg.metadata.dst_node_id]
        institution_id = context.node_config[protocol.INSTITUTION_ID]
        try:
            if (institution_id, msg.metadata.message_type) in self.failing:
                raise RuntimeError("simulated crash")
            reply = HANDLERS[msg.metadata.message_type](msg, context)
        except Exception as error:  # Flower turns handler exceptions into error replies
            reply = Message(Error(code=1, reason=str(error)), reply_to=msg)
        return None if institution_id in self.silent else reply
