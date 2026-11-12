"""Benchmark for PreTradeChecker on the Coinbase BTC/USD fixture.

The claim, written before any run: the counts per reason are identical in
the 3 sessions. There is no claim on the time; the JSON records the machine.
"""

import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

from dense_armor.core.finance.pretrade import PreTradeChecker

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "test/fixtures/finance/coinbase_btcusd_trades.csv"
OUT = ROOT / "benchmarks/finance_pretrade.json"


def _rows() -> list[dict[str, str]]:
    lines = FIXTURE.read_text().splitlines()
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","))) for ln in lines[1:] if ln]


def _session(rows: list[dict[str, str]], shift: float) -> tuple:
    c = PreTradeChecker(
        max_notional=200_000.0,
        max_size=0.1,
        price_band=0.0005,
        max_orders=500,
        window_s=60.0,
        duplicate_window_s=0.001,
    )
    prev = float(rows[0]["price"])
    reasons: Counter = Counter()
    times: list[float] = []
    for i, r in enumerate(rows):
        price = float(r["price"])
        size = float(r["size"])
        side = "buy" if r["side"] == "buy" else "sell"
        t = shift + i * 0.001
        t0 = time.perf_counter()
        d = c.check("acct", side, size, price, prev, t)
        times.append((time.perf_counter() - t0) * 1e6)
        reasons[d.reason] += 1
        prev = price
    return reasons, times


def main() -> None:
    rows = _rows()
    counts = []
    times_all: list[float] = []
    for s in range(3):
        c, ts = _session(rows, shift=float(s) * 10_000.0)
        counts.append(c)
        times_all.extend(ts)
    print("rows:", len(rows))
    for s, c in enumerate(counts, 1):
        print(f"  session {s}: {dict(sorted(c.items()))}")
    med = statistics.median(times_all)
    lo = min(times_all)
    hi = max(times_all)
    print(f"  time per call (us): median={med:.2f} min={lo:.2f} max={hi:.2f}")
    payload = {
        "backend": "python",
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or "unknown",
        },
        "sessions": [dict(sorted(c.items())) for c in counts],
        "time_per_call_us": {
            "median": med,
            "min": lo,
            "max": hi,
            "n": len(times_all),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
