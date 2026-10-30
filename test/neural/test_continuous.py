"""Item D of N05: checkpoint, protection, latency, and the online mode
of the N04 loop, with the classes of the library.

The task adds no code for these items: SafeEstimator, Protected,
AnomalyGate, HampelScorer, profile, Signal, RealtimePipeline are used
as they are."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import jax
import numpy as np

from dense_armor.roles import (
    AnomalyGate,
    Protected,
    RealtimePipeline,
    SafeEstimator,
    Signal,
    profile,
)
from dense_armor.utility.anomaly.filters import HampelScorer
from dense_armor.utility.neural.background import BackgroundLearner
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.online import OnlineNetRegressor
from dense_armor.utility.neural.trainer import Trainer
from dense_armor.utility.optim import SGD, Adam
from dense_armor.utility.stats.moments import RunningMoments


def _stream(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 2)).astype(np.float32)
    y = (2.0 * X[:, 0] - X[:, 1]).astype(np.float32)
    return X, y


def test_checkpoint_roundtrip_restores_adam_state():
    X, y = _stream(100)
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    a = OnlineNetRegressor(m1, MSE(), Adam(0.01), feature_keys=["a", "b"])
    sa = SafeEstimator(a)
    for i in range(50):
        sa.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, float(y[i]))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pkl"
        sa.save(path)
        m2 = mlp([2, 4, 1], jax.random.PRNGKey(0))
        b = OnlineNetRegressor(m2, MSE(), Adam(0.01), feature_keys=["a", "b"])
        sb = SafeEstimator(b)
        sb.restore(path)
        assert sb.model.n_seen_ == 50
        for p, q in zip(a.model_.params(), b.model_.params()):
            assert np.allclose(np.asarray(p.value), np.asarray(q.value))
        for i in range(50, 100):
            sa.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, float(y[i]))
            sb.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, float(y[i]))
        for p, q in zip(a.model_.params(), b.model_.params()):
            assert np.allclose(np.asarray(p.value), np.asarray(q.value), atol=1e-6)


def test_safe_estimator_guard_blocks_sample():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 0.3, (40, 1)).astype(np.float32)
    y = (X[:, 0] * 2.0).astype(np.float32)

    m = mlp([1, 4, 1], jax.random.PRNGKey(0))
    inner = OnlineNetRegressor(m, MSE(), SGD(0.1), feature_keys=["q"])
    safe = SafeEstimator(inner, guard=lambda x: abs(x["q"]) > 1e3, fallback=0.0)
    for i in range(20):
        safe.learn_one({"q": float(X[i, 0])}, float(y[i]))
    n_seen_before = inner.n_seen_
    w_before = [np.asarray(p.value).copy() for p in inner.model_.params()]
    out = safe.predict_one({"q": 1e7})
    assert out == 0.0
    assert inner.n_seen_ == n_seen_before
    for p, w in zip(inner.model_.params(), w_before):
        assert np.allclose(np.asarray(p.value), w)


def test_protected_with_anomaly_gate_blocks_outliers():
    hs = HampelScorer(radius=5, n_sigmas=3.0)
    gate = AnomalyGate(hs)
    for i in range(30):
        gate.learn_one({"q": float(i) * 0.01})
    m = mlp([1, 4, 1], jax.random.PRNGKey(0))
    inner = OnlineNetRegressor(m, MSE(), SGD(0.1), feature_keys=["q"])
    p = Protected(inner, gate)
    outlier_score = gate.score_one({"q": 1e7})
    normal_score = gate.score_one({"q": 0.15})
    print(f"outlier score={outlier_score:.3f} classify={gate.classify(outlier_score)}")
    print(f"normal score={normal_score:.3f} classify={gate.classify(normal_score)}")
    assert gate.classify(outlier_score) is True
    n_seen_before = inner.n_seen_
    _ = p.learn_one({"q": 1e7}, 0.0)
    assert inner.n_seen_ == n_seen_before
    _ = p.learn_one({"q": 0.15}, 0.30)
    assert inner.n_seen_ == n_seen_before + 1


def test_profile_online_regressor_fits():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    o = OnlineNetRegressor(m, MSE(), SGD(0.01), feature_keys=["a", "b"])
    signals = [
        Signal([float(i), float(-i)], ["a", "b"], ["rad", "rad"]) for i in range(200)
    ]
    r = profile(o, signals, warmup=10, method="predict_one")
    print(
        f"p50={r.p50_s * 1e3:.4f} ms p99={r.p99_s * 1e3:.4f} ms "
        f"budget={o.budget_s}s fits={r.fits}"
    )
    assert r.p99_s <= o.budget_s
    assert r.fits is True


def test_profile_hedge_fits():
    from dense_armor.utility.neural.hedge import HedgeNetClassifier

    h = HedgeNetClassifier(
        [2, 4], jax.random.PRNGKey(0), classes=[0, 1], feature_keys=["a", "b"]
    )
    signals = [
        Signal([float(i), float(-i)], ["a", "b"], ["rad", "rad"]) for i in range(200)
    ]
    r = profile(h, signals, warmup=10, method="predict_one")
    print(
        f"Hedge p50={r.p50_s * 1e3:.4f} ms p99={r.p99_s * 1e3:.4f} ms "
        f"budget={h.budget_s}s fits={r.fits}"
    )
    assert r.p50_s <= h.budget_s
    assert r.n_samples == 190
    assert isinstance(r.fits, bool)


def test_realtime_pipeline_headroom():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    o = OnlineNetRegressor(m, MSE(), SGD(0.01), feature_keys=["a", "b"])
    rm = RunningMoments()
    p = RealtimePipeline([o, rm], period_s=0.10, require_budget=False)
    h = p.headroom_s
    assert h is not None
    assert h > 0
    print(f"headroom_s={h:.4f} s, period=0.10 s")


def test_background_equals_trainer_batch_one():
    X, y = _stream(20)
    m1 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    t1 = Trainer(m1, MSE(), SGD(0.1))
    t1.fit(
        X,
        y.reshape(-1, 1),
        epochs=1,
        batch_size=1,
        key=jax.random.PRNGKey(0),
        shuffle=False,
    )
    w_trainer = [np.asarray(p.value).copy() for p in m1.params()]

    m2 = mlp([2, 4, 1], jax.random.PRNGKey(0))
    inner = OnlineNetRegressor(m2, MSE(), SGD(0.1), feature_keys=["a", "b"])
    b = BackgroundLearner(inner, maxsize=1000, sync_every=1)
    for i in range(20):
        b.learn_one({"a": float(X[i, 0]), "b": float(X[i, 1])}, float(y[i]))
    b.flush()
    w_bg = [np.asarray(p.value).copy() for p in b.served_.model_.params()]
    b.close()
    for p, q in zip(w_trainer, w_bg):
        assert np.allclose(p, q, atol=1e-6)


def test_background_latency_p99_under_2ms():
    m = mlp([2, 4, 1], jax.random.PRNGKey(0))
    inner = OnlineNetRegressor(m, MSE(), SGD(0.01), feature_keys=["a", "b"])
    b = BackgroundLearner(inner, maxsize=1000)
    lat = []
    for i in range(100):
        s = time.perf_counter()
        b.learn_one({"a": float(i), "b": float(-i)}, 0.0)
        lat.append((time.perf_counter() - s) * 1e3)
    p99 = float(np.percentile(lat, 99))
    b.flush()
    b.close()
    print(f"learn_one p99 through BackgroundLearner = {p99:.4f} ms")
    assert p99 < 2.0
