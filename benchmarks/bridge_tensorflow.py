"""Cost of the TensorFlow bridge against the libraries called directly.

Three measurements on the same model (a Sequential of Dense + Relu +
Dense), batch 256, float32:

- **jax direct**: the Dense-Armor model called on a jax array.
- **tf direct**: an equivalent ``tf.keras.Sequential`` on a tensor.
- **bridge**: ``KerasLayer`` wrapping the Dense-Armor model, called on
  a tensor (the forward crosses tf -> jax -> tf once).

7 runs each, median and IQR. The claim is declared in the first commit
and is not changed after the run: the bridge adds less than 20% of the
jax direct time on batch 256.

Usage:
    python benchmarks/bridge_tensorflow.py
    python benchmarks/bridge_tensorflow.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dense_armor.bridges.tensorflow import KerasLayer
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


def _tf_model():
    tf.random.set_seed(0)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_shape=(64,)),
            tf.keras.layers.Dense(32),
        ]
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
    tmodel = _tf_model()
    bridge = KerasLayer(model)
    x_j = jnp.zeros((batch, 64), dtype=jnp.float32)
    x_t = tf.zeros((batch, 64), dtype=tf.float32)

    t_jax = _time(lambda: model.forward(x_j, training=False), n_runs)
    t_tf = _time(lambda: tmodel(x_t), n_runs)
    t_bridge = _time(lambda: bridge(x_t), n_runs)

    out = {
        "backend": "cpu",
        "tensorflow": tf.__version__,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "batch": batch,
        "n_runs": n_runs,
        "jax_direct_ms": {
            "median": float(np.median(t_jax)),
            "iqr": float(np.percentile(t_jax, 75) - np.percentile(t_jax, 25)),
        },
        "tf_direct_ms": {
            "median": float(np.median(t_tf)),
            "iqr": float(np.percentile(t_tf, 75) - np.percentile(t_tf, 25)),
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
