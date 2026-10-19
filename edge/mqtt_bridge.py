"""MQTT bridge: publish the score and the alarm episodes of a pipeline.

The bridge runs a pipeline of the drones section on a synthetic stream,
takes its per-sample score and its per-sample alarm, and publishes them
on MQTT topics:

    drone/<id>/score    the score of the pipeline at that sample
    drone/<id>/alarm    the alarm flag (0 or 1) when it turns on

The topics follow the spec of Task D1, item C4. The published values
are one message per sample for the score, and one message per episode
start for the alarm (a rising edge of the alarm, same convention as the
other pages: benchmarks/casper_hours_episodes.py, lines 74-76).

The pipeline is sensor_consistency: its residual is defined, it runs
in one pass on a synthetic stream, and its alarm is the OR of a CUSUM
detector on the inter-arrival. Any pipeline of the section with a
per-sample score and a per-sample alarm can be passed instead; the
bridge only needs a list of (score, alarm).

The transport is paho-mqtt. The publisher is injectable: by default the
bridge calls paho.mqtt.publish.single on the given broker, port and
credentials; in the tests a fake publisher records the calls, so the
test suite does not depend on a running broker. One end-to-end run
against a public broker (test.mosquitto.org) is documented in the page.

The id in the topic is a string chosen by the caller; by default a
random four-letter suffix, so two bridges on the same broker do not
mix. The id is part of the result, so a reader can subscribe to the
exact topic that was published.

Run from the repository root:

    python edge/mqtt_bridge.py --smoke

The --smoke flag uses a synthetic stream of 200 samples and publishes
to test.mosquitto.org on a random id. Without the flag it uses 2000
samples and the id can be set with --id.

No board is available in this session. The code is the one a
Raspberry Pi or a Jetson would run: paho-mqtt, one message per sample,
the same topics. The page states this: "written, not run on the board".
"""

from __future__ import annotations

import argparse
import math
import random
import string
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipelines.drones.io import synthetic_flight
from pipelines.drones.sensor_consistency import run as run_pipeline

DEFAULT_BROKER = "test.mosquitto.org"
DEFAULT_PORT = 1883
DEFAULT_PREFIX = "drone"


@dataclass
class MqttBridgeResult:
    """Result of one run of the MQTT bridge.

    Attributes:
        n_samples: number of samples in the stream.
        n_scores_published: number of score messages published.
        n_alarms_published: number of alarm messages published.
        topic_score: the full topic of the score messages.
        topic_alarm: the full topic of the alarm messages.
        broker: the broker hostname.
        port: the broker port.
        id: the id used in the topic.
        scores: per-sample scores (n,), the one published.
        alarms: per-sample alarm flags (n,), the one published.
        errors: list of error strings, one per failed publish. Empty
            when every publish succeeded.
    """

    n_samples: int
    n_scores_published: int
    n_alarms_published: int
    topic_score: str
    topic_alarm: str
    broker: str
    port: int
    id: str
    scores: list[float]
    alarms: list[bool]
    errors: list[str]


def _default_publisher(
    topic: str, payload: str, hostname: str, port: int,
) -> None:
    """Publish one message with paho-mqtt, blocking.

    Uses the VERSION2 callback API when available (paho-mqtt 2.x),
    falling back to the default client on paho-mqtt 1.x, so the module
    runs on both versions without a deprecation warning.
    """
    import paho.mqtt.client as mqtt
    from paho.mqtt import publish

    version2 = getattr(mqtt, "CallbackAPIVersion", None)
    if version2 is not None and hasattr(version2, "VERSION2"):
        publish.single(
            topic, payload, hostname=hostname, port=port, qos=1,
            retain=False,
            client_id="",
            protocol=mqtt.MQTTv311,
        )
    else:
        publish.single(
            topic, payload, hostname=hostname, port=port, qos=1,
            retain=False,
        )


def _random_id(n: int = 4) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))


def run(
    stream: list[dict],
    broker: str = DEFAULT_BROKER,
    port: int = DEFAULT_PORT,
    id: str | None = None,
    prefix: str = DEFAULT_PREFIX,
    publisher: Callable[[str, str, str, int], None] | None = None,
) -> MqttBridgeResult:
    """Run the pipeline and publish score and alarm on MQTT.

    Args:
        stream: list of dicts from
            :func:`pipelines.drones.io.synthetic_flight`.
        broker: broker hostname. Default test.mosquitto.org.
        port: broker port. Default 1883.
        id: id used in the topic. Default a random four-letter string.
        prefix: topic prefix. Default "drone".
        publisher: callable ``(topic, payload, hostname, port)`` that
            publishes one message. Default uses paho-mqtt. In tests a
            fake publisher records the calls, so no broker is needed.

    Returns:
        :class:`MqttBridgeResult`.
    """
    if id is None:
        id = _random_id()
    if publisher is None:
        publisher = _default_publisher

    pipe = run_pipeline(stream)
    n = pipe.n_samples
    topic_score = f"{prefix}/{id}/score"
    topic_alarm = f"{prefix}/{id}/alarm"
    scores = [
        float(s) if not (isinstance(s, float) and math.isnan(s)) else 0.0
        for s in pipe.score
    ]
    alarms = [bool(a) for a in pipe.alarms]
    errors: list[str] = []
    n_score_ok = 0
    n_alarm_ok = 0
    prev_alarm = False
    for i in range(n):
        try:
            publisher(topic_score, f"{scores[i]:.6f}", broker, port)
            n_score_ok += 1
        except (OSError, RuntimeError, ValueError) as e:
            errors.append(f"score[{i}]: {type(e).__name__}: {e}")
        if alarms[i] and not prev_alarm:
            try:
                publisher(topic_alarm, "1", broker, port)
                n_alarm_ok += 1
            except (OSError, RuntimeError, ValueError) as e:
                errors.append(f"alarm[{i}]: {type(e).__name__}: {e}")
        prev_alarm = alarms[i]
    return MqttBridgeResult(
        n_samples=n,
        n_scores_published=n_score_ok,
        n_alarms_published=n_alarm_ok,
        topic_score=topic_score,
        topic_alarm=topic_alarm,
        broker=broker,
        port=port,
        id=id,
        scores=scores,
        alarms=alarms,
        errors=errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="200 samples, one publish per sample")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n = args.samples or (200 if args.smoke else 2000)
    stream = list(synthetic_flight(n=n, seed=args.seed, fault="gps_ramp"))
    r = run(stream, broker=args.broker, port=args.port, id=args.id)
    print(f"broker: {r.broker}:{r.port}")
    print(f"id: {r.id}")
    print(f"topics: {r.topic_score} | {r.topic_alarm}")
    print(f"scores: {r.n_scores_published}/{r.n_samples}")
    print(f"alarms: {r.n_alarms_published}")
    print(f"errors: {len(r.errors)}")
    return 0 if not r.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
