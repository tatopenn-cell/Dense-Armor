"""Comparison table on one regression stream with a sudden change.

Every learner sees the same 400-sample stream with two features. The
true weights are [1.0, -2.0] for the first 200 samples and [-1.0, 3.0]
for the last 200. Mean absolute error is reported on the first half
(before the change) and on the last 100 samples (after the change,
excluding the first 100 for the transient). The last column is the
median per-sample latency measured with time.perf_counter over the
whole stream.

Run with: pytest test/linear/test_comparison_table.py -s -v
"""

import time

import numpy as np

from dense_armor.utility.learn.online_dynamics import RecursiveLeastSquares
from dense_armor.utility.linear.bayes import BayesianLinearRegression
from dense_armor.utility.linear.glm import LinearRegression
from dense_armor.utility.optim import (
    SGD,
    AdaGrad,
    Adam,
    FTRLProximal,
    OnlineNewtonStep,
)


def _stream(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = np.empty(n)
    for i in range(n):
        w = np.array([1.0, -2.0]) if i < n // 2 else np.array([-1.0, 3.0])
        y[i] = float(w @ X[i]) + 0.05 * rng.normal()
    return X, y


def _run(est, X, y, keys=("a", "b")):
    preds = np.empty(len(y))
    times = np.empty(len(y))
    for i in range(len(y)):
        x = {keys[0]: float(X[i, 0]), keys[1]: float(X[i, 1])}
        t0 = time.perf_counter()
        preds[i] = float(est.predict_one(x))
        est.learn_one(x, float(y[i]))
        times[i] = time.perf_counter() - t0
    return preds, times


def _mae(preds, y, sl):
    return float(np.mean(np.abs(preds[sl] - y[sl])))


def test_comparison_table():
    X, y = _stream()
    pre = slice(0, 200)
    post = slice(300, 400)
    rows = []

    for opt, name in [
        (SGD(lr=0.05), "SGD(lr=0.05)"),
        (AdaGrad(lr=0.5), "AdaGrad(lr=0.5)"),
        (Adam(lr=0.05), "Adam(lr=0.05)"),
        (FTRLProximal(l2=0.1), "FTRLProximal(l2=0.1)"),
        (OnlineNewtonStep(), "OnlineNewtonStep"),
    ]:
        est = LinearRegression(optimizer=opt, intercept=True, feature_keys=["a", "b"])
        preds, times = _run(est, X, y)
        rows.append(
            (name, _mae(preds, y, pre), _mae(preds, y, post), float(np.median(times)))
        )

    est = BayesianLinearRegression(alpha=1e-6, beta=1e-6, feature_keys=["a", "b"])
    preds, times = _run(est, X, y)
    rows.append(
        (
            "BayesianLinearRegression",
            _mae(preds, y, pre),
            _mae(preds, y, post),
            float(np.median(times)),
        )
    )

    est = RecursiveLeastSquares(lam=0.99, delta=1e6, feature_keys=["a", "b"])
    preds, times = _run(est, X, y)
    rows.append(
        (
            "RecursiveLeastSquares(0.99)",
            _mae(preds, y, pre),
            _mae(preds, y, post),
            float(np.median(times)),
        )
    )

    print()
    print(f"{'learner':32s}  {'MAE pre':>9s}  {'MAE post':>9s}  {'median us':>10s}")
    print("-" * 68)
    for name, pre_mae, post_mae, t in rows:
        print(f"{name:32s}  {pre_mae:9.4f}  {post_mae:9.4f}  {t * 1e6:10.1f}")
    print()

    assert all(np.isfinite([r[1] for r in rows]))
    assert all(np.isfinite([r[2] for r in rows]))
