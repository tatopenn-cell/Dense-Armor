"""Video module benchmark: per-frame latency at 640x360.

Runs every module added in Task 21 on three synthetic streams (a smooth
motion, a sudden cut, a multi-channel thermal sequence) and prints the
median and the 99th percentile of the per-frame latency in
microseconds. The JSON next to the script carries the same numbers plus
the machine and the number of frames per stream.

The 1280x720 and 1920x1080 runs are not measured in the free Colab
session (2 cores at 1.6 GHz); only the 360p numbers are reported. The
sizes below are the only ones the script measures.

Run:

    python benchmarks/video_quality.py
"""

import json
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

N = 15
WARM = 3


def stream_smooth(H, W, seed=0):
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


def stream_cut(H, W, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(N):
        if i < N // 2:
            frames.append(rng.normal(0.4, 0.05, (H, W)).astype(np.float32))
        else:
            frames.append(rng.normal(0.7, 0.05, (H, W)).astype(np.float32))
    return np.stack(frames)


def stream_thermal(H, W, seed=0):
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
        ("FrameWindow", FrameWindow(n=5)),
        ("MovingObjectTracker", MovingObjectTracker(
            history=3, min_area=8, max_missing=3, n_sigmas=5.0,
            iou_threshold=0.2)),
        ("TrackToSignal", TrackToSignal()),
        ("FeatureToSignal", FeatureToSignal()),
        ("FrameJointFusion", FrameJointFusion(
            max_lag_s=0.1, window_s=0.5)),
        ("Registration", Registration()),
        ("Stabilizer", Stabilizer()),
        ("DifferenceImage", DifferenceImage(alpha=0.05)),
        ("DenseFlow", DenseFlow(alpha=1.0, n_iter=5)),
        ("MotionMask", MotionMask(n_sigmas=3.0, dilation=1)),
        ("FrameQuality", FrameQuality()),
        ("Sync", Sync(window=8, max_offset_s=0.05)),
        ("EventFrame", EventFrame(shape=(H, W), decay=0.9)),
        ("FalseColor", FalseColor(colormap="hot")),
        ("Overlay", Overlay()),
    ]


def run_stream(frames, H, W):
    rows = []
    # precalcolo dello sciame di flusso per MotionMask (una sola volta)
    flow = DenseFlow(alpha=1.0, n_iter=5)
    flow_prev = None
    flow_cache = []
    for i in range(N):
        f = frames[i]
        if f.ndim == 3:
            f_gray = f[..., 0]
        else:
            f_gray = f
        if flow_prev is None:
            flow_cache.append((np.zeros_like(f_gray), np.zeros_like(f_gray)))
        else:
            flow_prev_ = flow_prev
            out = flow.transform_one({"array": f_gray})
            flow_cache.append((out["u"], out["v"]))
        flow_prev = f_gray
    for modname, mod in make_modules(H, W):
        def one(i):
            frame = {"array": frames[i], "t": i * 0.033}
            if modname == "EventFrame":
                _ = mod.learn_one({"x": int(i % W), "y": int(i % H),
                                   "polarity": 1, "t": i * 0.033})
                _ = mod.transform_one(None)
            elif modname == "Overlay":
                _ = mod.transform_one({"array": frames[i]
                                       if frames[i].ndim == 3 else
                                       np.stack([frames[i]] * 3, axis=-1),
                                       "tracks": []})
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
            elif modname == "MotionMask":
                u, v = flow_cache[i]
                _ = mod.transform_one({"u": u, "v": v})
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
    out = {"machine": machine(), "n_frames": N, "warmup": WARM,
           "streams": {}, "runs": {}}
    H, W = 360, 640
    for name, gen in (("smooth", stream_smooth),
                      ("cut", stream_cut),
                      ("thermal", stream_thermal)):
        print(f"=== {name} {W}x{H} ===")
        frames = gen(H, W)
        rows = run_stream(frames, H, W)
        out["runs"][f"{name}_{W}x{H}"] = {
            m: {"median_us": md, "p99_us": p} for m, md, p in rows
        }
        for m, md, p in rows:
            print(f"  {m:22s}  median {md:9.1f}  p99 {p:9.1f} us")
    with open("benchmarks/video_quality.json", "w") as f:
        json.dump(out, f, indent=1)
    print()
    print("JSON: benchmarks/video_quality.json")


if __name__ == "__main__":
    main()
