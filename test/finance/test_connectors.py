"""Tests for the bounded stream and the broker adapters."""

import pytest

from pipelines.finance.connectors import (
    BoundedStream,
    KafkaConsumer,
    ZmqSubscriber,
)


def test_drop_oldest_keeps_the_newest():
    s = BoundedStream(capacity=2, policy="drop_oldest")
    for v in (1, 2, 3, 4):
        assert s.push(v) is True
    assert len(s) == 2
    assert s.counters.dropped_oldest == 2
    assert s.pop() == 3
    assert s.pop() == 4


def test_drop_newest_keeps_the_oldest():
    s = BoundedStream(capacity=2, policy="drop_newest")
    assert s.push(1) is True
    assert s.push(2) is True
    assert s.push(3) is False
    assert s.push(4) is False
    assert s.counters.dropped_newest == 2
    assert s.pop() == 1
    assert s.pop() == 2


def test_block_refuses_the_item():
    s = BoundedStream(capacity=1, policy="block")
    assert s.push("a") is True
    assert s.push("b") is False
    assert s.counters.blocked == 1
    assert len(s) == 1


def test_pop_on_empty_returns_none():
    s = BoundedStream(capacity=1)
    assert s.pop() is None
    assert s.counters.consumed == 0


def test_counters_count_pushed_and_consumed():
    s = BoundedStream(capacity=10)
    for v in range(5):
        s.push(v)
    for _ in range(3):
        s.pop()
    assert s.counters.pushed == 5
    assert s.counters.consumed == 3


def test_bad_capacity():
    with pytest.raises(ValueError, match="capacity"):
        BoundedStream(capacity=0)


def test_bad_policy():
    with pytest.raises(ValueError, match="policy"):
        BoundedStream(policy="bogus")


def test_zmq_publish_with_injected_client():
    class FakeClient:
        def __init__(self):
            self.calls = []
        def publish(self, topic, payload):
            self.calls.append((topic, payload))
    c = FakeClient()
    s = ZmqSubscriber(client=c, topic="t")
    s.publish(b"x")
    assert c.calls == [("t", b"x")]


def test_kafka_publish_with_injected_client():
    class FakeClient:
        def __init__(self):
            self.calls = []
        def publish(self, topic, payload):
            self.calls.append((topic, payload))
    c = FakeClient()
    s = KafkaConsumer(client=c, topic="k")
    s.publish(1)
    assert c.calls == [("k", 1)]


def test_missing_package_names_the_extra():
    s = ZmqSubscriber(client=None)
    s._package = "not_a_real_package_xyz"
    with pytest.raises(ImportError, match="finance-stream"):
        s.publish(1)


def test_zmq_client_is_built_and_sends():
    """The real zmq client is wired to Context, PUB and send_multipart."""
    import sys
    import types

    sent = []

    class FakeSock:
        def __init__(self):
            self.connected = None

        def connect(self, address):
            self.connected = address

        def send_multipart(self, parts):
            sent.append(parts)

        def close(self):
            pass

    class FakeContext:
        def socket(self, kind):
            return FakeSock()

        def term(self):
            pass

    fake = types.ModuleType("fake_zmq")
    fake.Context = FakeContext
    fake.PUB = 1
    sys.modules["fake_zmq"] = fake
    try:
        s = ZmqSubscriber(client=None, topic="t")
        s._package = "fake_zmq"
        s.publish({"x": 1})
        assert sent and sent[0][0] == b"t"
    finally:
        del sys.modules["fake_zmq"]


def test_kafka_client_is_built_and_sends():
    """The real Kafka client is wired to Producer.produce and poll."""
    import sys
    import types

    produced = []

    class FakeProducer:
        def __init__(self, conf):
            self.conf = conf

        def produce(self, topic, payload):
            produced.append((topic, payload))

        def poll(self, timeout):
            return 0

        def flush(self):
            pass

    fake = types.ModuleType("fake_confluent")
    fake.Producer = FakeProducer
    sys.modules["fake_confluent"] = fake
    try:
        s = KafkaConsumer(client=None, topic="k")
        s._package = "fake_confluent"
        s.publish(1)
        assert produced and produced[0][0] == "k"
    finally:
        del sys.modules["fake_confluent"]

def test_broker_adapter_is_abstract():
    from pipelines.finance.connectors import BrokerAdapter
    base = BrokerAdapter(client=None)
    with pytest.raises(NotImplementedError):
        base.publish(1)


def test_zmq_client_close_calls_close_and_term():
    import sys
    import types

    calls = []

    class FakeSock:
        def connect(self, address):
            pass

        def send_multipart(self, parts):
            pass

        def close(self):
            calls.append("sock.close")

    class FakeContext:
        def socket(self, kind):
            return FakeSock()

        def term(self):
            calls.append("ctx.term")

    fake = types.ModuleType("fake_zmq_close")
    fake.Context = FakeContext
    fake.PUB = 1
    sys.modules["fake_zmq_close"] = fake
    try:
        s = ZmqSubscriber(client=None, topic="t")
        s._package = "fake_zmq_close"
        c = s._resolve_client()
        c.close()
        assert calls == ["sock.close", "ctx.term"]
    finally:
        del sys.modules["fake_zmq_close"]


def test_kafka_client_flush_calls_flush():
    import sys
    import types

    flushed = []

    class FakeProducer:
        def __init__(self, conf):
            self.conf = conf

        def produce(self, topic, payload):
            pass

        def poll(self, timeout):
            return 0

        def flush(self):
            flushed.append(True)

    fake = types.ModuleType("fake_confluent_flush")
    fake.Producer = FakeProducer
    sys.modules["fake_confluent_flush"] = fake
    try:
        s = KafkaConsumer(client=None, topic="k")
        s._package = "fake_confluent_flush"
        c = s._resolve_client()
        c.flush()
        assert flushed == [True]
    finally:
        del sys.modules["fake_confluent_flush"]
