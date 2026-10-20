"""Connectors: what happens when the feed is faster than the pipeline?

A market feed can be faster than the pipeline that consumes it. The
queue between them has a fixed size, and when it is full something must
be dropped or the producer must wait. This module gives the queue with
three policies and the counters of what it dropped, and two adapters to
a message broker, imported lazily.

The page of this module says "written, not run against a broker": no
broker is in this environment, and the adapters are tested with a client
injected by the test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class StreamCounters:
    """What the queue did.

    Attributes:
        pushed: items the producer offered.
        consumed: items the consumer took.
        dropped_oldest: items dropped because the queue was full, the
            oldest first (``drop_oldest``).
        dropped_newest: items dropped because the queue was full, the
            new one (``drop_newest``).
        blocked: items the producer had to wait for (``block``).
    """

    pushed: int = 0
    consumed: int = 0
    dropped_oldest: int = 0
    dropped_newest: int = 0
    blocked: int = 0


class BoundedStream:
    """A fixed-size queue with a drop or block policy.

    Args:
        capacity: maximum number of items in the queue. Must be >= 1.
        policy: ``"drop_oldest"``, ``"drop_newest"`` or ``"block"``.

    Raises:
        ValueError: on a non-positive ``capacity`` or an unknown policy.
    """

    def __init__(self, capacity: int = 100, policy: str = "drop_oldest") -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if policy not in ("drop_oldest", "drop_newest", "block"):
            raise ValueError(
                f"policy must be 'drop_oldest', 'drop_newest' or 'block', "
                f"got {policy!r}")
        self.capacity = int(capacity)
        self.policy = policy
        self._q: deque = deque()
        self.counters = StreamCounters()

    def push(self, item) -> bool:
        """Offer one item. Returns True if the item entered the queue."""
        self.counters.pushed += 1
        if len(self._q) < self.capacity:
            self._q.append(item)
            return True
        if self.policy == "drop_oldest":
            self._q.popleft()
            self._q.append(item)
            self.counters.dropped_oldest += 1
            return True
        if self.policy == "drop_newest":
            self.counters.dropped_newest += 1
            return False
        self.counters.blocked += 1
        return False

    def pop(self):
        """Take one item, or None when the queue is empty."""
        if not self._q:
            return None
        self.counters.consumed += 1
        return self._q.popleft()

    def __len__(self) -> int:
        return len(self._q)


class BrokerAdapter:
    """Base for the broker adapters, with a lazy import of the client.

    Args:
        client: a client object with a ``publish(topic, payload)``
            method, or ``None`` to import the real one.
        topic: the topic to publish on.
        extra: the name of the optional extra, used in the ImportError.
        package: the module to import when ``client`` is None.
    """

    def __init__(self, client=None, topic: str = "ticks",
                 extra: str = "finance-stream",
                 package: str = "") -> None:
        self.topic = topic
        self._client = client
        self._extra = extra
        self._package = package

    def _resolve_client(self):
        if self._client is not None:
            return self._client
        try:
            module = __import__(self._package, fromlist=["_"])
        except ImportError as exc:
            raise ImportError(
                f"{self._package} is not installed; install the "
                f"[{self._extra}] extra") from exc
        return module

    def publish(self, payload) -> None:
        client = self._resolve_client()
        client.publish(self.topic, payload)


class ZmqSubscriber(BrokerAdapter):
    """A ZeroMQ publisher, written and not run against a broker."""

    def __init__(self, client=None, topic: str = "ticks") -> None:
        super().__init__(client=client, topic=topic,
                         extra="finance-stream", package="zmq")


class KafkaConsumer(BrokerAdapter):
    """A Kafka producer, written and not run against a broker."""

    def __init__(self, client=None, topic: str = "ticks") -> None:
        super().__init__(client=client, topic=topic,
                         extra="finance-stream", package="confluent_kafka")
