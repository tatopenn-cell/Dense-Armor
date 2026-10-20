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


def test_missing_package_message_names_the_module():
    s = KafkaConsumer(client=None)
    s._package = "not_a_real_package_xyz"
    with pytest.raises(ImportError, match="not_a_real_package_xyz"):
        s.publish(1)


def test_publish_with_real_module():
    """The resolved module is returned when the import succeeds."""
    import sys
    import types

    class FakeModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.calls = []

        def publish(self, topic, payload):
            self.calls.append((topic, payload))

    fake = FakeModule("fake_broker_xyz")
    sys.modules["fake_broker_xyz"] = fake
    try:
        s = ZmqSubscriber(client=None, topic="t")
        s._package = "fake_broker_xyz"
        s.publish("x")
        assert fake.calls == [("t", "x")]
    finally:
        del sys.modules["fake_broker_xyz"]
