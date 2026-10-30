"""Cost of the torch bridge against the libraries called directly.

Three measurements on the same model (a Sequential of Dense + Relu +
Dense), batch 256, float32:

- **jax direct**: the Dense-Armor model called on a jax array.
- **torch direct**: an equivalent ``torch.nn.Sequential`` on a tensor.
- **bridge**: ``TorchModule`` wrapping the Dense-Armor model, called on
  a tensor (the forward crosses torch -> jax -> torch once).

7 runs each, median and IQR. The claim is declared in the first commit
and is not changed after the run: the bridge adds less than 20% of the
jax direct time on batch 256.

Usage:
    python benchmarks/bridge_torch.py
    python benchmarks/bridge_torch.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_armor.bridges.torch import TorchModule
from dense_armor.utility.neural import Dense, Sequential
from dense_armor.utility.neural.activations import Relu


def _model():
    return Sequential(
        [
            Dense(64, 128, key=jax.random.PRNGKey(0)),
            Relu(),
            Dense(128, 32, key=jax.random.PRNGKey(1)),
        ]
    )


def _torch_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 32)
    )


def _time(fn, n_runs, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def main(smoke: bool) -> dict:
    batch = 32 if smoke else 256
    n_runs = 3 if smoke else 7
    model = _model()
    tmodel = _torch_model()
    bridge = TorchModule(model)
    x_j = jnp.zeros((batch, 64), dtype=jnp.float32)
    x_t = torch.zeros(batch, 64, dtype=torch.float32)

    t_jax = _time(lambda: model.forward(x_j, training=False), n_runs)
    t_torch = _time(lambda: tmodel(x_t), n_runs)
    t_bridge = _time(lambda: bridge(x_t), n_runs)

    out = {
        "backend": "cpu",
        "torch": torch.__version__,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "batch": batch,
        "n_runs": n_runs,
        "jax_direct_ms": {
            "median": float(np.median(t_jax)),
            "iqr": float(np.percentile(t_jax, 75) - np.percentile(t_jax, 25)),
        },
        "torch_direct_ms": {
            "median": float(np.median(t_torch)),
            "iqr": float(np.percentile(t_torch, 75) - np.percentile(t_torch, 25)),
        },
        "bridge_ms": {
            "median": float(np.median(t_bridge)),
            "iqr": float(np.percentile(t_bridge, 75) - np.percentile(t_bridge, 25)),
        },
    }
    overhead = (out["bridge_ms"]["median"] - out["jax_direct_ms"]["median"]) / max(
        out["jax_direct_ms"]["median"], 1e-9
    )
    out["overhead_fraction"] = float(overhead)
    out["claim_20pct_holds"] = bool(overhead < 0.20)

    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(args.smoke)
