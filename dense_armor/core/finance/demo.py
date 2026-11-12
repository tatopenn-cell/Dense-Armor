"""Demo command: synthetic feed, checker, log, one JSON line on stdout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from dense_armor.core.finance.config import load_config
from dense_armor.core.finance.decision_log import DecisionLog
from dense_armor.core.finance.feed import iter_itch
from dense_armor.core.finance.ring import RingBuffer
from dense_armor.core.finance.schedule import twap_schedule
from dense_armor.core.finance.synthetic import synthetic_itch


def _write_feed(path: Path, n: int, seed: int) -> int:
    data = synthetic_itch(n, seed)
    path.write_bytes(data)
    return len(data)


def _run(config_path: str, n: int, seed: int) -> tuple[int, dict]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return 2, {"error": f"--n must be a non-negative integer, got {n}", "ok": False}
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        return 2, {
            "error": f"--seed must be a non-negative integer, got {seed}",
            "ok": False,
        }

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return 2, {"error": str(e), "ok": False}

    if config.ring.width != 4:
        return 2, {"error": "ring.width must be 4 for the demo", "ok": False}

    if config.paths.emergency_stop.exists():
        return 3, {"error": "emergency stop", "ok": False}

    if config.paths.log.exists():
        return 2, {"error": "log exists", "ok": False}

    t0 = time.perf_counter()

    _write_feed(config.paths.feed, n, seed)

    slices = twap_schedule(config.schedule.total, config.schedule.n)
    schedule_sum = float(slices.sum())

    ring: RingBuffer = config.make_ring()
    checker = config.make_checker()
    dlog = DecisionLog(config.paths.log)
    out_row: np.ndarray = np.zeros(4, dtype=np.float64)
    ref_price: dict[str, float] = {}

    n_messages = 0
    n_orders = 0
    counts: dict[str, int] = {}

    for m in iter_itch(config.paths.feed):
        n_messages += 1
        if m["type"] not in ("A", "F"):
            continue
        n_orders += 1
        t_seconds = m["timestamp"] / 1e9
        ring.push(np.array([m["order_ref"], m["shares"], m["price"], t_seconds]))
        ring.pop_into(out_row)
        stock = m["stock"]
        if stock not in ref_price:
            ref_price[stock] = m["price"]
        side = "buy" if m["side"] == "B" else "sell"
        decision = checker.check(
            account=stock,
            side=side,
            size=m["shares"],
            price=m["price"],
            ref_price=ref_price[stock],
            t=t_seconds,
        )
        counts[decision.reason] = counts.get(decision.reason, 0) + 1
        dlog.append(
            t_seconds,
            {
                "approved": bool(decision.approved),
                "order_ref": m["order_ref"],
                "reason": decision.reason,
                "stock": stock,
            },
        )

    elapsed_s = time.perf_counter() - t0

    payload = {
        "ok": True,
        "counts": counts,
        "elapsed_s": float(elapsed_s),
        "log": {
            "last_hash": dlog.last_hash,
            "records": len(dlog),
            "verified": dlog.verify(),
        },
        "n_messages": n_messages,
        "n_orders": n_orders,
        "python": sys.version.split()[0],
        "schedule_sum": schedule_sum,
    }
    return 0, payload


def main(argv: list[str] | None = None) -> int:
    """Run the demo command and return the exit code.

    The command runs a synthetic ITCH feed through the pre-trade checker and
    records every decision in the log. It prints one JSON object on stdout
    and nothing else.

    Args:
        argv: the argument list, or ``None`` to read :data:`sys.argv`.

    Returns:
        The exit code: 0 for a clean run, 2 for a configuration error or a
        configuration the demo does not accept, 3 if the stop file exists.

    Examples:
        >>> import contextlib, io
        >>> buf = io.StringIO()
        >>> with contextlib.redirect_stdout(buf):
        ...     code = main(["--config", "/nonexistent/demo.toml"])
        >>> code
        2
    """
    parser = argparse.ArgumentParser(prog="python -m dense_armor.core.finance.demo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    code, payload = _run(args.config, args.n, args.seed)
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
