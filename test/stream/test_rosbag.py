"""Tests for iter_rosbag.

The test writes a small ROS 1 bag with the rosbags writer and reads it
back with the library reader; skipped if rosbags is not installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from dense_armor.utility.stream import iter_rosbag

ROS1_WRITER_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32",
    reason="rosbags builds ROS 1 message definitions with PosixPath, unavailable on Windows",
)


def _write_bag(path: Path) -> None:
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    JointState = ts.types["sensor_msgs/msg/JointState"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]

    def _serialize(msg):
        if hasattr(ts, "serialize_ros1"):
            return ts.serialize_ros1(msg, "sensor_msgs/msg/JointState")
        return ts.serialize_cdr(msg, "sensor_msgs/msg/JointState")

    with Writer(str(path)) as w:
        conn = w.add_connection(
            "/joint_states",
            "sensor_msgs/msg/JointState",
            typestore=ts,
        )
        for i in range(3):
            msg = JointState(
                header=Header(
                    seq=i,
                    stamp=Time(sec=i, nanosec=0),
                    frame_id="base",
                ),
                name=["j0", "j1"],
                position=np.array([0.1 + i, 0.2 + i], dtype=np.float64),
                velocity=np.array([0.01 + i, 0.02 + i], dtype=np.float64),
                effort=np.array([0.5 + i, 0.6 + i], dtype=np.float64),
            )
            w.write(conn, i * 1_000_000_000, _serialize(msg))


@ROS1_WRITER_WINDOWS
def test_iter_rosbag_joint_state(tmp_path):
    pytest.importorskip("rosbags")
    path = tmp_path / "test.bag"
    _write_bag(path)
    out = list(iter_rosbag(path))
    assert len(out) == 3
    sig0, y0 = out[0]
    assert y0 is None
    assert sig0.t == 0.0
    assert sig0["q_j0"] == 0.1
    assert sig0["tau_j0"] == 0.5
    assert sig0.units[sig0.names.index("q_j0")] == "rad"
    sig2, _ = out[2]
    assert sig2.t == 2.0


@ROS1_WRITER_WINDOWS
def test_iter_rosbag_topic_filter(tmp_path):
    pytest.importorskip("rosbags")
    path = tmp_path / "test.bag"
    _write_bag(path)
    assert list(iter_rosbag(path, topics=["/nonexistent"])) == []
    assert len(list(iter_rosbag(path, topics=["/joint_states"]))) == 3


def test_iter_rosbag_missing_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("rosbags"):
            raise ImportError("no rosbags")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import importlib

    mod = importlib.reload(importlib.import_module("dense_armor.utility.stream.rosbag"))
    with pytest.raises(ImportError):
        list(mod.iter_rosbag("/anything"))
