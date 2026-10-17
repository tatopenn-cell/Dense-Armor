"""Hotspots: cProfile of five models, top 10 functions by cumulative time.

Models: ``AdaptiveRandomForestClassifier``, ``MondrianForestRegressor``,
``FIMTDDRegressor``, ``GlobalMotion`` at 640x360, ``DenseFlow``.
2000 samples (15 frames for the vision models). No code change: the
tables are the delivery.

Command: ``python benchmarks/hotspots.py``. Smoke: ``--smoke``.
"""

import argparse
import cProfile
import io
import pstats
from pathlib import Path

import numpy as np


def _profile(label, run, top=10):
    """Run ``run()`` under cProfile and print the top functions.

    Args:
        label: name printed above the table.
        run: a zero-arg callable that performs the work.
        top: how many functions to print.
    """
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(top)
    print(f"\n=== {label} ===")
    for line in s.getvalue().splitlines():
        if line.strip():
            print(f"  {line}")


def _harf(n, seed=0):
    from dense_armor.utility.forest.arf import AdaptiveRandomForestClassifier
    rng = np.random.default_rng(seed)
    m = AdaptiveRandomForestClassifier(n_models=5, seed=seed)
    for i in range(n):
        x = {f"x{k}": float(rng.normal(0, 1)) for k in range(4)}
        y = int(x["x0"] > 0)
        m.learn_one(x, y)
        m.predict_one(x)


def _hmon(n, seed=0):
    from dense_armor.utility.tree.mondrian import MondrianForestRegressor
    rng = np.random.default_rng(seed)
    m = MondrianForestRegressor(n_trees=5, seed=seed)
    for i in range(n):
        x = {f"x{k}": float(rng.normal(0, 1)) for k in range(4)}
        y = float(x["x0"] + 0.5 * x["x1"])
        m.learn_one(x, y)
        m.predict_one(x)


def _hfimt(n, seed=0):
    from dense_armor.utility.tree.fimt import FIMTDDRegressor
    rng = np.random.default_rng(seed)
    m = FIMTDDRegressor()
    for i in range(n):
        x = {f"x{k}": float(rng.normal(0, 1)) for k in range(4)}
        y = float(x["x0"] + 0.5 * x["x1"])
        try:
            m.learn_one(x, y)
            m.predict_one(x)
        except AttributeError:
            pass


def _hgm(n_frames, seed=0):
    from dense_armor.utility.vision.motion import GlobalMotion
    rng = np.random.default_rng(seed)
    m = GlobalMotion()
    for i in range(n_frames):
        a = rng.normal(0, 1, (360, 640)).astype(np.float32)
        b = np.roll(a, (2, 1), axis=(0, 1))
        m.transform_one({"array": b, "prev": a})


def _hdf(n_frames, seed=0):
    from dense_armor.utility.vision.dense import DenseFlow
    rng = np.random.default_rng(seed)
    m = DenseFlow(alpha=1.0, n_iter=10)
    for i in range(n_frames):
        a = rng.normal(0, 1, (360, 640)).astype(np.float32)
        b = np.roll(a, (1, 1), axis=(0, 1))
        m.learn_one({"array": a})
        m.transform_one({"array": b})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    n = 200 if a.smoke else a.samples
    nf = 2 if a.smoke else a.frames

    _profile("AdaptiveRandomForestClassifier", lambda: _harf(n))
    _profile("MondrianForestRegressor", lambda: _hmon(n))
    _profile("FIMTDDRegressor (broken on master, see taskZ3 delivery)",
             lambda: _hfimt(n))
    _profile("GlobalMotion 640x360", lambda: _hgm(nf))
    _profile("DenseFlow 640x360", lambda: _hdf(nf))


if __name__ == "__main__":
    main()
