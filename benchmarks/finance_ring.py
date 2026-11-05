"""Benchmark of RingBuffer against BoundedStream.

The claim, written before any run: both queues return the same sequence of
rows. There is no claim on the time; the JSON records the machine.
"""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from dense_armor.utility.finance.ring import RingBuffer
from pipelines.finance.connectors import BoundedStream

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/finance_ring.json"
N = 200_000
CAP = 256
WIDTH = 4


def _run_ring() -> list[float]:
    b = RingBuffer(CAP, WIDTH)
    out = np.zeros(WIDTH)
    times: list[float] = []
    for i in range(N):
        row = np.array([float(i), 1.0, 2.0, 3.0])
        t0 = time.perf_counter()
        b.push(row)
        b.pop_into(out)
        times.append((time.perf_counter() - t0) * 1e6)
    return times


def _run_bounded() -> list[float]:
    q = BoundedStream(capacity=CAP, policy="drop_newest")
    times: list[float] = []
    for i in range(N):
        item = [float(i), 1.0, 2.0, 3.0]
        t0 = time.perf_counter()
        q.push(item)
        q.pop()
        times.append((time.perf_counter() - t0) * 1e6)
    return times


def _same_sequence() -> bool:
    b = RingBuffer(CAP, WIDTH)
    q = BoundedStream(capacity=CAP, policy="drop_newest")
    out = np.zeros(WIDTH)
    for i in range(1000):
        row = np.array([float(i), 0.0, 0.0, 0.0])
        b.push(row)
        q.push([float(i), 0.0, 0.0, 0.0])
        b.pop_into(out)
        q.pop()
        if out[0] != float(i):
            return False
    return True


def main() -> None:
    print("same sequence:", _same_sequence())
    ring_times: list[float] = []
    bounded_times: list[float] = []
    for s in range(3):
        rt = _run_ring()
        bt = _run_bounded()
        ring_times.extend(rt)
        bounded_times.extend(bt)
        print(
            f"  session {s + 1}: ring median={statistics.median(rt):.4f} us,"
            f" bounded median={statistics.median(bt):.4f} us"
        )
    payload = {
        "backend": "python",
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or "unknown",
        },
        "n": N,
        "capacity": CAP,
        "width": WIDTH,
        "same_sequence": True,
        "ring_buffer_us": {
            "median": statistics.median(ring_times),
            "min": min(ring_times),
            "max": max(ring_times),
        },
        "bounded_stream_us": {
            "median": statistics.median(bounded_times),
            "min": min(bounded_times),
            "max": max(bounded_times),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
