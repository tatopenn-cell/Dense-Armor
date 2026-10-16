"""Video module benchmark: latency per frame at 640x360 and 1920x1080.

Runs every module of ``dense_armor.utility.vision`` that was added in
Task 21 on three synthetic streams (a smooth motion, a sudden cut, a
multi-channel thermal sequence), at two resolutions, and prints the
median and the 99th percentile of the per-frame latency in
microseconds. The JSON next to the script carries the same numbers
plus the machine.

Run:

    python benchmarks/video_quality.py

The modules the script measures: FrameNormalizer, MotionEnergy,
GlobalMotion, SceneChange, FrameAnomaly, FrameWindow, MovingObjectTracker,
TrackToSignal, FeatureToSignal, FrameJointFusion, Registration,
Stabilizer, DifferenceImage, DenseFlow, MotionMask, FrameQuality, Sync,
EventFrame, FalseColor, Overlay. The file reader is excluded because it
needs a decoder, not a frame in memory.
"""

import json
import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dense_armor.utility.vision.adapters import (  # noqa: E402
    FeatureToSignal,
    TrackToSignal,
)
from dense_armor.utility.vision.dense import DenseFlow, MotionMask  # noqa: E402
from dense_armor.utility.vision.fusion import FrameJointFusion  # noqa: E402
from dense_armor.utility.vision.motion import GlobalMotion, MotionEnergy  # noqa: E402
from dense_armor.utility.vision.quality import (  # noqa: E402
    EventFrame,
    FrameQuality,
    Sync,
)
from dense_armor.utility.vision.registration import (  # noqa: E402
    DifferenceImage,
    Registration,
    Stabilizer,
)
from dense_armor.utility.vision.robustness import FrameNormalizer  # noqa: E402
from dense_armor.utility.vision.scene import FrameAnomaly, SceneChange  # noqa: E402
from dense_armor.utility.vision.streams import ArrayStream  # noqa: E402
from dense_armor.utility.vision.tracking import MovingObjectTracker  # noqa: E402
from dense_armor.utility.vision.visualize import (  # noqa: E402
    FalseColor,
    Overlay,
)
from dense_armor.utility.vision.window import FrameWindow  # noqa: E402

N = 60
WARM = 5


def stream_a(H, W, seed=0):
    """Smooth motion: a disk on a noisy background."""
    rng = np.random.default_rng(seed)
    frames = []
    pos = np.array([H // 3, W // 3], dtype=float)
    vel = np.array([1.5, 1.0])
    for _ in range(N):
        f = rng.normal(0.1, 0.01, (H, W)).astype(np.float32)
        y, x = int(pos[0]), int(pos[1])
        f[max(0, y):y + 6, max(0, x):x + 6] = 1.0
        frames.append(f)
        pos = pos + vel
        if pos[0] > H - 6:
            pos[0] = 0.0
        if pos[1] > W - 6:
            pos[1] = 0.0
    return np.stack(frames)


def stream_b(H, W, seed=0):
    """Sudden cut in the middle."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(N):
        if i < N // 2:
            frames.append(rng.normal(0.4, 0.05, (H, W)).astype(np.float32))
        else:
            frames.append(rng.normal(0.7, 0.05, (H, W)).astype(np.float32))
    return np.stack(frames)


def stream_c(H, W, seed=0):
    """Multi-channel thermal: 4 channels, no RGB assumption."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(N):
        cube = rng.normal(300.0, 5.0, (H, W, 4)).astype(np.float32)
        cube[H // 3: H // 3 + 5, W // 3: W // 3 + 5, :] += 50.0
        frames.append(cube)
    return np.stack(frames)


def measure(fn):
    ns = []
    for i in range(N):
        t0 = time.perf_counter_ns()
        fn(i)
        ns.append(time.perf_counter_ns() - t0)
    a = np.array(ns[WARM:], dtype=float) / 1e3
    return float(np.median(a)), float(np.percentile(a, 99))


def make_modules(H, W):
    return [
        ("FrameNormalizer", FrameNormalizer()),
        ("MotionEnergy", MotionEnergy(blur=1, downsample=2)),
        ("GlobalMotion", GlobalMotion()),
        ("SceneChange", SceneChange()),
        ("FrameAnomaly", FrameAnomaly()),
        ("FrameWindow", FrameWindow(n=8)),
        ("MovingObjectTracker", MovingObjectTracker(history=5, min_area=8, max_missing=5)),
        ("TrackToSignal", TrackToSignal()),
        ("FeatureToSignal", FeatureToSignal()),
        ("FrameJointFusion", FrameJointFusion(max_lag_s=0.1, window_s=0.5)),
        ("Registration", Registration()),
        ("Stabilizer", Stabilizer()),
        ("DifferenceImage", DifferenceImage(alpha=0.05)),
        ("DenseFlow", DenseFlow(alpha=1.0, n_iter=20)),
        ("MotionMask", MotionMask(n_sigmas=3.0, dilation=1)),
        ("FrameQuality", FrameQuality()),
        ("Sync", Sync(window=32, max_offset_s=0.1)),
        ("EventFrame", EventFrame(shape=(H, W), decay=0.9)),
        ("FalseColor", FalseColor(colormap="hot")),
        ("Overlay", Overlay()),
    ]


def run_stream(name, frames, H, W):
    rows = []
    for modname, mod in make_modules(H, W):
        def one(i):
            frame = {"array": frames[i], "t": i * 0.033}
            if modname == "EventFrame":
                _ = mod.transform_one(None)
                _ = mod.learn_one({"x": int(i % W), "y": int(i % H),
                                   "polarity": 1, "t": i * 0.033})
            elif modname == "Overlay":
                _ = mod.transform_one(frame)
            elif modname == "Sync":
                _ = mod.add_a(float(frames[i].mean()), i * 0.033)
                _ = mod.add_b(float(frames[i].mean()), i * 0.033)
                _ = mod.transform_one(None)
            elif modname == "FrameJointFusion":
                _ = mod.add_video_velocity(float(frames[i].mean()), i * 0.033)
                _ = mod.add_joint_speed(float(frames[i].mean()), i * 0.033)
            elif modname == "TrackToSignal":
                tr = {"tracks": [{"id": 1, "box": (2, 2, 8, 8), "missing": 0}]}
                _ = mod.learn_one(tr)
                _ = mod.transform_one(None, t=i * 0.033)
            elif modname in ("Stabilizer", "DifferenceImage"):
                _ = mod.learn_one(frame)
                _ = mod.transform_one(frame)
            else:
                _ = mod.transform_one(frame)
                _ = mod.learn_one(frame)
        try:
            med, p99 = measure(one)
        except Exception as e:
            med, p99 = float("nan"), float("nan")
            print(f"  {modname}: EXC {type(e).__name__}: {str(e)[:80]}")
        rows.append((modname, med, p99))
    return rows


def machine():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(),
            "python": sys.version.split()[0], "platform": platform.platform()}


def main():
    print("machine:", machine())
    out = {"machine": machine(), "n_frames": N, "warmup": WARM, "runs": {}}
    for res in [(640, 360), (1920, 1080)]:
        H, W = res
        for stream_name, gen in (("smooth", stream_a),
                                 ("cut", stream_b),
                                 ("thermal", stream_c)):
            print(f"=== {stream_name} {W}x{H} ===")
            frames = gen(H, W)
            rows = run_stream(stream_name, frames, H, W)
            out["runs"][f"{stream_name}_{W}x{H}"] = {
                name: {"median_us": m, "p99_us": p}
                for name, m, p in rows
            }
            for name, m, p in rows:
                print(f"  {name:22s}  median {m:9.1f}  p99 {p:9.1f} us")
    with open("benchmarks/video_quality.json", "w") as f:
        json.dump(out, f, indent=1)
    print()
    print("JSON: benchmarks/video_quality.json")


if __name__ == "__main__":
    main()
