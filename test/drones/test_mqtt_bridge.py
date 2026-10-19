"""Tests for edge/mqtt_bridge.py (task D1, item C4).

The tests use a fake publisher, so the suite does not depend on a
running MQTT broker. One end-to-end run against a public broker
(test.mosquitto.org) is documented in the page, not in the tests.
"""

import sys

import sys

import numpy as np
import pytest

from edge.mqtt_bridge import (
    DEFAULT_BROKER,
    DEFAULT_PORT,
    DEFAULT_PREFIX,
    MqttBridgeResult,
    _random_id,
    run,
)
from pipelines.drones.io import synthetic_flight


def _fake_publisher(calls: list):
    def publish(topic: str, payload: str, hostname: str, port: int) -> None:
        calls.append((topic, payload, hostname, port))
    return publish


def _stream(n=200, seed=0, fault=None):
    return list(synthetic_flight(n=n, seed=seed, fault=fault))


def test_run_publishes_all_scores():
    calls = []
    stream = _stream(n=200)
    r = run(stream, id="test", publisher=_fake_publisher(calls))
    assert isinstance(r, MqttBridgeResult)
    assert r.n_samples == 200
    assert r.n_scores_published == 200
    assert r.id == "test"
    assert r.topic_score == f"{DEFAULT_PREFIX}/test/score"
    assert r.topic_alarm == f"{DEFAULT_PREFIX}/test/alarm"


def test_run_alarms_only_on_rising_edges():
    calls = []
    stream = _stream(n=200, fault="gps_ramp")
    r = run(stream, id="test", publisher=_fake_publisher(calls))
    n_score_msgs = sum(1 for c in calls if c[0] == r.topic_score)
    n_alarm_msgs = sum(1 for c in calls if c[0] == r.topic_alarm)
    assert n_score_msgs == 200
    assert n_alarm_msgs == r.n_alarms_published
    assert n_alarm_msgs >= 1


def test_no_fault_no_alarms():
    calls = []
    stream = _stream(n=100)
    r = run(stream, id="test", publisher=_fake_publisher(calls))
    assert r.n_alarms_published == 0


def test_random_id_length():
    for _ in range(5):
        assert len(_random_id()) == 4
        assert len(_random_id(8)) == 8


def test_defaults():
    assert DEFAULT_BROKER == "test.mosquitto.org"
    assert DEFAULT_PORT == 1883
    assert DEFAULT_PREFIX == "drone"


def test_errors_recorded():
    def failing_pub(topic, payload, hostname, port):
        raise RuntimeError("broker down")
    stream = _stream(n=10, fault="gps_ramp")
    r = run(stream, id="test", publisher=failing_pub)
    assert len(r.errors) >= 10
    assert r.n_scores_published == 0
    assert r.n_alarms_published == 0


def test_scores_match_pipeline():
    from pipelines.drones.sensor_consistency import run as pipeline_run
    stream = _stream(n=100, fault="gps_ramp")
    calls = []
    r = run(stream, id="test", publisher=_fake_publisher(calls))
    pipe = pipeline_run(stream)
    for i in (10, 50, 99):
        if pipe.score[i] == pipe.score[i]:
            assert abs(r.scores[i] - float(pipe.score[i])) < 1e-9


def test_topic_structure():
    calls = []
    stream = _stream(n=10)
    r = run(stream, id="abc123", publisher=_fake_publisher(calls), prefix="fleet")
    assert r.topic_score == "fleet/abc123/score"
    assert r.topic_alarm == "fleet/abc123/alarm"
    for c in calls:
        assert c[2] == DEFAULT_BROKER
        assert c[3] == DEFAULT_PORT

def test_main_runs(monkeypatch):
    """main() with --smoke, using a fake publisher via monkeypatch."""
    from edge import mqtt_bridge as mod
    calls = []
    def fake_default(topic, payload, hostname, port):
        calls.append((topic, payload))
    monkeypatch.setattr(mod, "_default_publisher", fake_default)
    monkeypatch.setattr(sys, "argv", ["mqtt_bridge", "--smoke", "--id", "test"])
    rc = mod.main()
    assert rc == 0
    assert len(calls) > 0


def test_main_errors_returns_1(monkeypatch):
    from edge import mqtt_bridge as mod
    def failing(topic, payload, hostname, port):
        raise RuntimeError("broker down")
    monkeypatch.setattr(mod, "_default_publisher", failing)
    monkeypatch.setattr(sys, "argv", ["mqtt_bridge", "--smoke", "--id", "test"])
    rc = mod.main()
    assert rc == 1


def test_default_publisher_paho1x(monkeypatch):
    """Fallback branch: paho-mqtt 1.x has no CallbackAPIVersion."""
    from edge import mqtt_bridge as mod
    import paho.mqtt.client as mqtt_real
    calls = []
    def fake_single(topic, payload, hostname, port, qos, retain, **kw):
        calls.append((topic, payload, hostname, port))
    import paho.mqtt.publish as publish_real
    monkeypatch.setattr(publish_real, "single", fake_single)
    monkeypatch.delattr(mqtt_real, "CallbackAPIVersion", raising=False)
    mod._default_publisher("t/s", "1.0", "h", 1883)
    assert len(calls) == 1


def test_default_publisher_paho2x(monkeypatch):
    """VERSION2 branch: paho-mqtt 2.x has CallbackAPIVersion."""
    from edge import mqtt_bridge as mod
    calls = []
    def fake_single(topic, payload, hostname, port, qos, retain, **kw):
        calls.append((topic, payload, hostname, port))
    import paho.mqtt.publish as publish_real
    monkeypatch.setattr(publish_real, "single", fake_single)
    mod._default_publisher("t/s", "1.0", "h", 1883)
    assert len(calls) == 1

