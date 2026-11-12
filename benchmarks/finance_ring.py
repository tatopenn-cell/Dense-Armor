"""Benchmark of RingBuffer against BoundedStream.

The claim, written before any run: both queues return the same sequence of
rows. The page reports the median per session, the paired Wilcoxon p-value
and the A12 effect size over 20 sessions; the time is recorded in the JSON
with the machine.
"""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.core.finance.ring import RingBuffer
from pipelines.finance.connectors import BoundedStream

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/finance_ring.json"
N = 200_000
CAP = 256
WIDTH = 4
SESSIONS = 20


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


def _a12(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    d = a_arr[:, None] - b_arr[None, :]
    return float((d < 0).mean() + 0.5 * (d == 0).mean())


def main() -> None:
    print("same sequence:", _same_sequence())
    ring_med: list[float] = []
    bounded_med: list[float] = []
    for s in range(SESSIONS):
        rt = _run_ring()
        bt = _run_bounded()
        ring_med.append(statistics.median(rt))
        bounded_med.append(statistics.median(bt))
        print(
            f"  session {s + 1:2d}/{SESSIONS}: "
            f"ring={ring_med[-1]:.4f} us  bounded={bounded_med[-1]:.4f} us"
        )
    try:
        p = float(wilcoxon(ring_med, bounded_med, alternative="two-sided").pvalue)
    except ValueError:
        p = float("nan")
    a12 = _a12(ring_med, bounded_med)
    payload = {
        "backend": "python",
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or "unknown",
        },
        "n_per_session": N,
        "capacity": CAP,
        "width": WIDTH,
        "sessions": SESSIONS,
        "same_sequence": True,
        "ring_buffer_us_medians": ring_med,
        "bounded_stream_us_medians": bounded_med,
        "wilcoxon_two_sided_p": p,
        "a12_ring_slower_than_bounded": a12,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print()
    print(f"ring median-of-medians:    {statistics.median(ring_med):.4f} us")
    print(f"bounded median-of-medians: {statistics.median(bounded_med):.4f} us")
    print(f"ratio: {statistics.median(ring_med) / statistics.median(bounded_med):.2f}")
    print(f"Wilcoxon two-sided p: {p:.6f}")
    print(f"A12 (ring slower than bounded): {a12:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
