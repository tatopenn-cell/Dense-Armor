"""Cost of ``generate`` with and without the static key/value cache.

A ``RotaryDecoder`` is run for ``--new`` tokens from a prompt of 8 tokens,
once through the cache (``init_static_cache`` and ``step_compiled``) and
once through a wrapper that exposes only ``forward``, which re-runs the
whole prefix at every token. Each path is timed ``--repeats`` times after
one warm-up call; the JSON has the first call, the median and the range of
the repeats, and whether the two paths give the same tokens.

Usage:
    python benchmarks/neural/kv_cache.py --json benchmarks/neural/kv_cache_cpu.json
    python benchmarks/neural/kv_cache.py --smoke --json out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dense_armor.utility.deep.neural.generate import generate  # noqa: E402
from dense_armor.utility.deep.neural.rotary_decoder import rotary_decoder  # noqa: E402


class _ForwardOnly:
    """The same model seen through ``forward`` only: no cache."""

    def __init__(self, model):
        self.model = model
        self.max_len = model.max_len

    def forward(self, ids):
        return self.model.forward(ids, training=False)


def _time_path(model, prompt, new, repeats):
    t0 = time.perf_counter()
    generate(model, prompt, max_new=1)
    first = time.perf_counter() - t0
    runs = []
    out = None
    for _ in range(repeats):
        t = time.perf_counter()
        out = generate(model, prompt, max_new=new)
        runs.append(time.perf_counter() - t)
    return {
        "first_call_s": first,
        "median_s": statistics.median(runs),
        "range_s": [min(runs), max(runs)],
        "runs_s": runs,
    }, out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--new", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    cfg = dict(vocab=65, d_model=64, n_heads=4, n_layers=2, max_len=256)
    new, repeats = args.new, args.repeats
    if args.smoke:
        cfg.update(d_model=16, n_heads=2, n_layers=1, max_len=32)
        new, repeats = 8, 1
    model = rotary_decoder(**cfg, key=jax.random.PRNGKey(0))
    prompt = jnp.arange(8, dtype=jnp.int32)
    cached, out_c = _time_path(model, prompt, new, repeats)
    plain, out_p = _time_path(_ForwardOnly(model), prompt, new, repeats)
    row = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "config": cfg,
        "n_params": sum(int(p.value.size) for p in model.params()),
        "prompt_len": int(prompt.shape[0]),
        "new_tokens": new,
        "repeats": repeats,
        "cache": cached,
        "no_cache": plain,
        "speedup_median": plain["median_s"] / cached["median_s"],
        "same_tokens": bool((out_c == out_p).all()),
    }
    text = json.dumps(row, indent=2)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return row


if __name__ == "__main__":
    main(sys.argv[1:])
