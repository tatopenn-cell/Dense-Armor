"""Comparison table on a stream with one sudden change.

Every new learner of the task, against the library learner it competes
with: a single Hoeffding tree. Median of five seeds, time per sample.
"""

import time

import numpy as np

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.ensemble import (
    OnlineBaggingClassifier,
    OnlineBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


def _stream(seed: int, n: int = 2000):
    ds = DriftStream(n_samples=n, n_features=4, n_drifts=1, kind="sudden", seed=seed)
    return [(x.to_dict(), y) for x, y in ds.stream()]


def _run(make, data):
    model = make()
    correct = 0
    n = 0
    t0 = time.perf_counter()
    for x, y in data:
        p = model.predict_one(x)
        if p is not None and p == y:
            correct += 1
        model.learn_one(x, y)
        n += 1
    dt = time.perf_counter() - t0
    return correct / n, dt / n * 1e6


def _report(name: str, make, n_seeds: int = 5):
    accs = []
    times = []
    for seed in range(n_seeds):
        data = _stream(seed)
        acc, us = _run(make, data)
        accs.append(acc)
        times.append(us)
    print(
        f"  {name:24s} acc median {np.median(accs):.4f}  "
        f"time/sample {np.median(times):8.2f} us"
    )
    return float(np.median(accs)), float(np.median(times))


def test_comparison_table(capsys):
    """Print the table; the test only checks it is well formed."""
    print()
    print("=== comparison, 5 seeds, sudden drift, 2000 samples ===")
    _report("HoeffdingTree (single)", lambda: HoeffdingTreeClassifier(grace_period=50))
    _report(
        "OnlineBagging (3)",
        lambda: OnlineBaggingClassifier(
            lambda: HoeffdingTreeClassifier(grace_period=50), n_models=3, seed=0
        ),
    )
    _report(
        "OnlineBoosting (3)",
        lambda: OnlineBoostingClassifier(
            lambda: HoeffdingTreeClassifier(grace_period=50), n_models=3, seed=0
        ),
    )
    _report(
        "Voting (3, hard)",
        lambda: VotingClassifier(
            [lambda: HoeffdingTreeClassifier(grace_period=50)] * 3, mode="hard"
        ),
    )
    _report(
        "Stacking (2+meta)",
        lambda: StackingClassifier(
            [lambda: HoeffdingTreeClassifier(grace_period=50)] * 2,
            lambda: HoeffdingTreeClassifier(grace_period=50),
        ),
    )
