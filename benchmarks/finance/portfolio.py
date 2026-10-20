"""Write benchmarks/finance/portfolio.json from the four fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.finance.portfolio import run_on_fixtures

FIXTURES = ROOT / "test" / "fixtures" / "finance"
OUT = ROOT / "benchmarks" / "finance" / "portfolio.json"

r = run_on_fixtures(FIXTURES)
payload = {
    "dataset": "four fixtures, daily log returns",
    "tickers": ["AAPL", "GSPC", "MSFT", "TSLA"],
    "n_samples": r.n_samples,
    "alpha": r.alpha,
    "beta": r.beta,
    "weights_min_var": r.weights_min_var.tolist(),
    "weights_inv_vol": r.weights_inv_vol.tolist(),
    "realised_vol_min_var": r.realised_vol_min_var,
    "realised_vol_inv_vol": r.realised_vol_inv_vol,
    "realised_vol_equal": r.realised_vol_equal,
    "turnover_min_var": r.turnover_min_var,
}
OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
