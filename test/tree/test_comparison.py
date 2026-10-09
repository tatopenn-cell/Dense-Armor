import time

import numpy as np

from dense_armor.utility.metrics import Accuracy, MeanAbsoluteError
from dense_armor.utility.tree.adaptive import HoeffdingAdaptiveTreeClassifier
from dense_armor.utility.tree.efdt import HoeffdingAnytimeTreeClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
from dense_armor.utility.tree.mondrian import (
    MondrianForestClassifier,
    MondrianForestRegressor,
)
from dense_armor.utility.tree.sgt import SGTClassifier, SGTRegressor


class _PeriodicDetector:
    def __init__(self, period: int = 300) -> None:
        self.period = period
        self.n_ = 0
        self.drift_detected = False

    def update(self, x, t=None) -> "_PeriodicDetector":
        self.n_ += 1
        self.drift_detected = self.n_ % self.period == 0
        return self


def _reg_stream(rng, n, drift_at=None):
    for i in range(n):
        x = float(rng.uniform(-1.0, 1.0))
        y = float(np.sin(3.0 * x))
        if drift_at is not None and i >= drift_at:
            y = float(np.cos(3.0 * x))
        yield {"x": x}, y


def _cls_stream(rng, n, drift_at=None):
    for i in range(n):
        y = int(rng.random() < 0.5)
        c = 0.0 if y == 0 else 5.0
        if drift_at is not None and i >= drift_at:
            c += 4.0
        yield {"x0": float(rng.normal(c, 0.4))}, y


def _score_reg(model, stream):
    mae = MeanAbsoluteError()
    t0 = time.perf_counter()
    for x, y in stream:
        pred = model.predict_one(x)
        if pred is None:
            pred = 0.0
        mae.update(y, pred)
        model.learn_one(x, y=y)
    dt = time.perf_counter() - t0
    return mae.get(), dt / max(1, len(stream))


def _score_cls(model, stream):
    acc = Accuracy()
    t0 = time.perf_counter()
    for x, y in stream:
        pred = model.predict_one(x)
        if pred is None:
            pred = -1
        acc.update(y, pred)
        model.learn_one(x, y=y)
    dt = time.perf_counter() - t0
    return acc.get(), dt / max(1, len(stream))


def test_comparison_table():
    rng = np.random.default_rng(0)
    n = 3000
    reg_stat = list(_reg_stream(rng, n))
    reg_drift = list(_reg_stream(rng, n, drift_at=n // 2))
    cls_stat = list(_cls_stream(rng, n))
    cls_drift = list(_cls_stream(rng, n, drift_at=n // 2))

    rows = []
    for name, ctor in (
        (
            "SGTRegressor",
            lambda: SGTRegressor(min_samples_split=50, n_bins=32, bin_samples=200),
        ),
        (
            "MondrianForestRegressor",
            lambda: MondrianForestRegressor(n_trees=10, seed=0, min_samples_split=5),
        ),
    ):
        m, dt = _score_reg(ctor(), list(reg_stat))
        rows.append((name, "stationary", "MAE", f"{m:.4f}", f"{dt * 1000:.3f}"))
        m, dt = _score_reg(ctor(), list(reg_drift))
        rows.append((name, "drift", "MAE", f"{m:.4f}", f"{dt * 1000:.3f}"))

    for name, ctor in (
        ("HoeffdingTreeClassifier", lambda: HoeffdingTreeClassifier(grace_period=50)),
        (
            "HoeffdingAnytimeTreeClassifier",
            lambda: HoeffdingAnytimeTreeClassifier(grace_period=50),
        ),
        (
            "HoeffdingAdaptiveTreeClassifier",
            lambda: HoeffdingAdaptiveTreeClassifier(
                grace_period=50, drift_detector=_PeriodicDetector(period=500)
            ),
        ),
        (
            "SGTClassifier",
            lambda: SGTClassifier(min_samples_split=50, n_bins=32, bin_samples=200),
        ),
        (
            "MondrianForestClassifier",
            lambda: MondrianForestClassifier(n_trees=10, seed=0, min_samples_split=5),
        ),
    ):
        a, dt = _score_cls(ctor(), list(cls_stat))
        rows.append((name, "stationary", "Acc", f"{a:.4f}", f"{dt * 1000:.3f}"))
        a, dt = _score_cls(ctor(), list(cls_drift))
        rows.append((name, "drift", "Acc", f"{a:.4f}", f"{dt * 1000:.3f}"))

    print(f"{'model':<35} {'stream':<11} {'metric':<7} {'value':>8} {'ms/sample':>10}")
    print("-" * 74)
    for r in rows:
        print(f"{r[0]:<35} {r[1]:<11} {r[2]:<7} {r[3]:>8} {r[4]:>10}")
