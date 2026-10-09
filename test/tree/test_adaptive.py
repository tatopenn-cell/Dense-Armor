import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.drift.adwin import ADWIN
from dense_armor.utility.metrics import Accuracy
from dense_armor.utility.tree.adaptive import HoeffdingAdaptiveTreeClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier


class _PeriodicDetector:
    """Minimal detector that fires every ``period`` updates."""

    def __init__(self, period: int = 200) -> None:
        self.period = period
        self.n_ = 0
        self.drift_detected = False

    def update(self, x, t=None) -> "_PeriodicDetector":
        self.n_ += 1
        self.drift_detected = self.n_ % self.period == 0
        return self


def _boundary_flip_stream(rng, n_pre, n_post, noise=0.05):
    for _ in range(n_pre):
        x = float(rng.normal(0.0, 1.0))
        y = 1 if x > 0.0 else 0
        if rng.random() < noise:
            y = 1 - y
        yield {"x0": x}, y
    for _ in range(n_post):
        x = float(rng.normal(0.0, 1.0))
        y = 1 if x < 0.0 else 0
        if rng.random() < noise:
            y = 1 - y
        yield {"x0": x}, y


def test_hat_beats_hoeffding_on_boundary_flip():
    rng = np.random.default_rng(0)
    pre = list(_boundary_flip_stream(rng, 2000, 0))
    post = list(_boundary_flip_stream(rng, 0, 3000))

    ht = HoeffdingTreeClassifier(grace_period=50, leaf="majority", max_depth=1)
    for x, y in pre:
        ht.learn_one(x, y=y)
    acc_ht = Accuracy()
    for x, y in post:
        pred = ht.predict_one(x)
        if pred is None:
            pred = -1
        acc_ht.update(y, pred)
        ht.learn_one(x, y=y)
    ht_acc = acc_ht.get()

    hat = HoeffdingAdaptiveTreeClassifier(
        grace_period=50,
        leaf="majority",
        max_depth=1,
        drift_detector=ADWIN(delta=0.002),
        delta_alt=0.05,
        kappa_alt=30,
        error_alpha=0.01,
    )
    for x, y in pre:
        hat.learn_one(x, y=y)
    acc_hat = Accuracy()
    for x, y in post:
        pred = hat.predict_one(x)
        if pred is None:
            pred = -1
        acc_hat.update(y, pred)
        hat.learn_one(x, y=y)
    hat_acc = acc_hat.get()

    print(f"HT:  acc={ht_acc:.4f}")
    print(
        f"HAT: acc={hat_acc:.4f}, n_drifts={hat.n_drifts_}, "
        f"n_alt_starts={hat.n_alt_starts_}, n_swaps={hat.n_swaps_}"
    )
    assert hat.n_drifts_ > 0
    assert hat.n_alt_starts_ > 0
    assert hat_acc > ht_acc + 0.05


def test_hat_swap_logic_with_periodic_detector():
    rng = np.random.default_rng(1)
    pre = list(_boundary_flip_stream(rng, 1000, 0))
    post = list(_boundary_flip_stream(rng, 0, 1000))
    hat = HoeffdingAdaptiveTreeClassifier(
        grace_period=50,
        leaf="majority",
        max_depth=1,
        drift_detector=_PeriodicDetector(period=200),
        delta_alt=0.05,
        kappa_alt=20,
        error_alpha=0.01,
    )
    for x, y in pre:
        hat.learn_one(x, y=y)
    for x, y in post:
        hat.learn_one(x, y=y)
    assert hat.n_drifts_ > 0
    assert hat.n_alt_starts_ > 0


def test_hat_smoke():
    rng = np.random.default_rng(2)
    h = HoeffdingAdaptiveTreeClassifier(grace_period=30, drift_detector=ADWIN())
    for _ in range(400):
        y = int(rng.random() < 0.5)
        c = 0.0 if y == 0 else 5.0
        d = {"x0": float(rng.normal(c, 0.4))}
        h.learn_one(d, y=y)
    p0 = h.predict_one({"x0": 0.0})
    p1 = h.predict_one({"x0": 5.0})
    assert p0 == 0 and p1 == 1


def test_hat_nan_counted():
    h = HoeffdingAdaptiveTreeClassifier(grace_period=20, drift_detector=ADWIN())
    h.learn_one({"x0": 0.0}, y=0)
    h.learn_one({"x0": float("nan")}, y=1)
    assert h.n_missing == 1


def test_check_estimator_hat():
    check_estimator(
        HoeffdingAdaptiveTreeClassifier(grace_period=20, drift_detector=ADWIN())
    )
