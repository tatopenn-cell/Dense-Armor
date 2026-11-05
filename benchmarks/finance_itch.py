"""Benchmark for the ITCH 5.0 decoder.

The claim, written before any run: 100,000 messages are decoded without
an exception in each session. There is no claim on the time.
"""

import json
import platform
import statistics
import time
from pathlib import Path

from dense_armor.utility.finance.itch import decode_message

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/finance_itch.json"
N = 100_000


def _msg_a() -> bytes:
    return (
        b"A"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (0).to_bytes(6, "big")
        + (7).to_bytes(8, "big")
        + b"B"
        + (100).to_bytes(4, "big")
        + b"AAPL    "
        + (1234567).to_bytes(4, "big")
    )


def _run() -> list[float]:
    m = _msg_a()
    times: list[float] = []
    for _ in range(N):
        t0 = time.perf_counter()
        d = decode_message(m)
        times.append((time.perf_counter() - t0) * 1e6)
        assert d["type"] == "A"
    return times


def main() -> None:
    all_times: list[float] = []
    for s in range(3):
        ts = _run()
        all_times.extend(ts)
        print(f"  session {s + 1}: {N} messages decoded, no exception")
    med = statistics.median(all_times)
    lo = min(all_times)
    hi = max(all_times)
    print(f"  time per message (us): median={med:.4f} min={lo:.4f} max={hi:.4f}")
    payload = {
        "backend": "python",
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or "unknown",
        },
        "sessions": [{"messages": N, "exception": None} for _ in range(3)],
        "time_per_message_us": {
            "median": med,
            "min": lo,
            "max": hi,
            "n": len(all_times),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
