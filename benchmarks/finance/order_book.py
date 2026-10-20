"""Write benchmarks/finance/order_book.json from a synthetic book."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.finance.order_book import (
    order_flow_imbalance,
    regress,
    synthetic_book,
)

OUT = ROOT / "benchmarks" / "finance" / "order_book.json"
SLOPE = 2e-4
NOISE = 1e-3

book, ofi_true = synthetic_book(n=2000, seed=0, slope=SLOPE, noise=NOISE)
res = regress(book, true_slope=SLOPE)
ofi = order_flow_imbalance(book)

payload = {
    "dataset": "synthetic book, planted linear relation",
    "n_events": res.n_events,
    "planted_slope": SLOPE,
    "noise": NOISE,
    "estimated_slope": res.slope,
    "intercept": res.intercept,
    "r2": res.r2,
    "ofi_median": float(abs(ofi).mean()),
    "ofi_max": float(abs(ofi).max()),
}
OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
