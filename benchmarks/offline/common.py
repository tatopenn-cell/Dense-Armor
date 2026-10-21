"""Shared helpers of the classic against online comparison.

A case is a function ``seed -> (classic, online, kind)``: the two outputs on
the same synthetic data and ``kind``, ``"abs"`` for the largest absolute
difference or ``"rel"`` for the largest relative difference (the error of a
sketch against the exact value). ``run`` runs the cases on 10 seeds
(3 with ``--smoke``) and writes ``benchmarks/offline/<family>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NOT_COMPUTED = (
    "not computed: the classic and the online value are two outputs of the same "
    "formula on the same data, so the table gives their difference and not a "
    "ranking of the two"
)


def feed(online, rows):
    """Call ``learn_one`` of ``online`` on every dict of ``rows``; return it."""
    for r in rows:
        online.learn_one(r)
    return online


def diff(classic, online, kind: str) -> float:
    """Largest absolute or relative difference between two outputs."""
    a = np.asarray(classic, dtype=float)
    b = np.asarray(online, dtype=float)
    d = np.abs(a - b)
    if kind == "rel":
        d = d / np.maximum(np.abs(a), 1e-12)
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else float("nan")


def run(
    family: str, cases: dict[str, Callable], why: dict[str, str], argv=None
) -> dict:
    """Run the cases, print the table and write the JSON of the family."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    seeds = 3 if args.smoke else 10
    result: dict = {
        "family": family,
        "seeds": seeds,
        "synthetic": True,
        "wilcoxon_a12": NOT_COMPUTED,
        "classes": {},
    }
    for name, fn in cases.items():
        vals, kind = [], "abs"
        for s in range(seeds):
            c, o, kind = fn(s)
            vals.append(diff(c, o, kind))
        v = np.asarray(vals)
        result["classes"][name] = {
            "kind": kind,
            "median": float(np.nanmedian(v)),
            "p90": float(np.nanquantile(v, 0.9)),
            "max": float(np.nanmax(v)),
            "why": why.get(name, ""),
        }
    out = (
        Path(args.out)
        if args.out
        else ROOT / "benchmarks" / "offline" / f"{family}.json"
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{'class':34s} {'kind':4s} {'median':>10s} {'p90':>10s} {'max':>10s}")
    for name, r in result["classes"].items():
        print(
            f"{name:34s} {r['kind']:4s} {r['median']:10.3g} {r['p90']:10.3g} {r['max']:10.3g}"
        )
    return result
