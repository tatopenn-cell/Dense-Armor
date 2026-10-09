"""Read ROS 1 and ROS 2 bags as a stream of :class:`Signal` samples.

The pure Python package ``rosbags`` reads both formats; it is an
optional dependency, installed with the extra ``[ros]``, and imported
inside :func:`iter_rosbag`. Every ``sensor_msgs/msg/JointState`` message
becomes one :class:`Signal` with channels ``q_<joint>``,
``qd_<joint>``, ``tau_<joint>``.

Examples:
    >>> from dense_armor.utility.stream import iter_rosbag
    >>> stream = iter_rosbag("/path/to/bag")  # doctest: +SKIP
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dense_armor.roles import Signal

_JOINT_STATE = "sensor_msgs/msg/JointState"


def _load_rosbags() -> Any:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ImportError(
            "iter_rosbag requires the rosbags package. Install it with "
            "`pip install dense-armor[ros]`."
        ) from exc
    return AnyReader


def iter_rosbag(
    path: str | Path,
    topics: list[str] | None = None,
) -> Iterator[tuple[Signal, Any]]:
    """Iterate a ROS 1 or ROS 2 bag as ``(Signal, y)`` pairs.

    Args:
        path: ROS 1 ``.bag`` file, or ROS 2 bag directory.
        topics: topic names to read. ``None`` = every topic with a
            ``sensor_msgs/msg/JointState`` message.

    Yields:
        ``(signal, None)`` pairs, in message order.

    Raises:
        ImportError: if ``rosbags`` is not installed.
    """
    AnyReader = _load_rosbags()
    path = Path(path)
    with AnyReader([path]) as reader:
        if topics is None:
            conns = [c for c in reader.connections if c.msgtype == _JOINT_STATE]
        else:
            wanted = set(topics)
            conns = [
                c
                for c in reader.connections
                if c.msgtype == _JOINT_STATE and c.topic in wanted
            ]
        if not conns:
            return
        for conn, ts_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            name = list(msg.name)
            pos = list(msg.position)
            vel = list(msg.velocity)
            eff = list(msg.effort)
            vals: dict[str, float] = {}
            units: dict[str, str] = {}
            for j, jname in enumerate(name):
                if j < len(pos):
                    vals[f"q_{jname}"] = float(pos[j])
                    units[f"q_{jname}"] = "rad"
                if j < len(vel):
                    vals[f"qd_{jname}"] = float(vel[j])
                    units[f"qd_{jname}"] = "rad/s"
                if j < len(eff):
                    vals[f"tau_{jname}"] = float(eff[j])
                    units[f"tau_{jname}"] = "N*m"
            if not vals:
                continue
            t_s = float(ts_ns) * 1e-9
            sig = Signal.from_dict(vals, units=units, t=t_s)
            yield sig, None
