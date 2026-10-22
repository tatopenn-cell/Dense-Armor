"""Tests for background.py."""

from __future__ import annotations

import threading
import time

import jax
import numpy as np
import pytest

from dense_armor.utility.neural.background import BackgroundLearner
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.models import mlp
from dense_armor.utility.neural.online import OnlineNetRegressor
from dense_armor.utility.optim import SGD


class SlowRegressor:
    def __init__(self) -> None:
        self.n = 0

    def learn_one(self, x, y, t=None):
        time.sleep(0.02)
        self.n += 1
        return self

    def predict_one(self, x, t=None):
        return self.n


class BrokenRegressor:
    def __init__(self) -> None:
        self.n = 0

    def learn_one(self, x, y, t=None):
        raise RuntimeError("boom")

    def predict_one(self, x, t=None):
        return self.n


def _p99(values):
    return float(np.percentile(values, 99))


def test_main_not_blocked():
    b = BackgroundLearner(SlowRegressor(), maxsize=1000)
    t0 = time.perf_counter()
    lat = []
    for i in range(100):
        s = time.perf_counter()
        b.learn_one({"a": float(i)}, 0.0)
        lat.append((time.perf_counter() - s) * 1e3)
    total = time.perf_counter() - t0
    b.flush()
    b.close()
    print(f"learn_one p99={_p99(lat):.3f} ms, total={total:.3f} s")
    assert _p99(lat) < 2.0
    assert total < 0.5

    # sequenziale: >= 2 s
    s = SlowRegressor()
    t0 = time.perf_counter()
    for i in range(100):
        s.learn_one({"a": float(i)}, 0.0)
    total_seq = time.perf_counter() - t0
    print(f"sequential: {total_seq:.3f} s")
    assert total_seq >= 2.0


def test_predict_not_blocked():
    b = BackgroundLearner(SlowRegressor(), maxsize=1000)
    for i in range(50):
        b.learn_one({"a": float(i)}, 0.0)
    lat = []
    for i in range(100):
        s = time.perf_counter()
        b.predict_one({"a": float(i)})
        lat.append((time.perf_counter() - s) * 1e3)
    p99 = _p99(lat)
    print(f"predict_one p99={p99:.3f} ms")
    b.flush()
    b.close()
    assert p99 < 2.0


def test_same_result_as_sequential():
    X = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    y = (X * 2.0 + 0.1).astype(np.float32)
    dicts = [{"a": float(X[i])} for i in range(20)]

    # sequenziale
    m1 = mlp([1, 4, 1], jax.random.PRNGKey(0))
    seq = OnlineNetRegressor(m1, MSE(), SGD(0.1), feature_keys=["a"])
    for d, yi in zip(dicts, y):
        seq.learn_one(d, float(yi))

    # background
    m2 = mlp([1, 4, 1], jax.random.PRNGKey(0))
    inner = OnlineNetRegressor(m2, MSE(), SGD(0.1), feature_keys=["a"])
    b = BackgroundLearner(inner, maxsize=1000, sync_every=1)
    for d, yi in zip(dicts, y):
        b.learn_one(d, float(yi))
    b.flush()
    w_seq = [np.asarray(p.value).copy() for p in seq.model_.params()]
    w_bg = [np.asarray(p.value).copy() for p in b.served_.model_.params()]
    diff = max(float(np.max(np.abs(a - c))) for a, c in zip(w_seq, w_bg))
    print(f"weights diff: {diff:.2e}")
    b.close()
    assert diff < 1e-6


def test_drop_rule():
    b = BackgroundLearner(SlowRegressor(), maxsize=2)
    for i in range(50):
        b.learn_one({"a": float(i)}, 0.0)
    b.flush()
    print(f"n_sent={b.n_sent_} n_learned={b.n_learned_} n_dropped={b.n_dropped_}")
    assert b.n_dropped_ > 0
    assert b.n_learned_ + b.n_dropped_ == b.n_sent_
    b.close()


def test_close_joins_thread():
    before = threading.active_count()
    b = BackgroundLearner(SlowRegressor(), maxsize=100)
    b.learn_one({"a": 1.0}, 0.0)
    b.flush()
    b.close()
    time.sleep(0.05)
    after = threading.active_count()
    assert after == before
    b.close()


def test_context_manager():
    before = threading.active_count()
    with BackgroundLearner(SlowRegressor(), maxsize=100) as b:
        b.learn_one({"a": 1.0}, 0.0)
        b.flush()
    time.sleep(0.05)
    assert threading.active_count() == before


def test_error_propagates():
    b = BackgroundLearner(BrokenRegressor(), maxsize=100)
    b.learn_one({"a": 1.0}, 0.0)
    with pytest.raises(RuntimeError, match="boom"):
        b.flush(timeout=2.0)
    b.close()


def test_bound_past_the_bound():
    n = 50
    b_small = BackgroundLearner(SlowRegressor(), maxsize=2)
    for i in range(n):
        b_small.learn_one({"a": float(i)}, 0.0)
    b_small.flush()
    learned_small = b_small.n_learned_
    b_small.close()

    m1 = mlp([1, 4, 1], jax.random.PRNGKey(0))
    inner_small = OnlineNetRegressor(m1, MSE(), SGD(0.1), feature_keys=["a"])
    b_small2 = BackgroundLearner(inner_small, maxsize=2)
    for i in range(n):
        b_small2.learn_one({"a": float(i)}, 0.0)
    b_small2.flush()
    w_small = [np.asarray(p.value).copy() for p in b_small2.served_.model_.params()]
    b_small2.close()

    m2 = mlp([1, 4, 1], jax.random.PRNGKey(0))
    inner_big = OnlineNetRegressor(m2, MSE(), SGD(0.1), feature_keys=["a"])
    b_big = BackgroundLearner(inner_big, maxsize=1000)
    for i in range(n):
        b_big.learn_one({"a": float(i)}, 0.0)
    b_big.flush()
    w_big = [np.asarray(p.value).copy() for p in b_big.served_.model_.params()]
    b_big.close()

    diff = max(float(np.max(np.abs(a - c))) for a, c in zip(w_small, w_big))
    print(
        f"learned_small={learned_small} learned_big={b_big.n_learned_} "
        f"weights diff={diff:.4f}"
    )
    assert learned_small < b_big.n_learned_
    assert diff > 1e-3


def test_value_errors():
    with pytest.raises(ValueError, match="maxsize must be at least 1"):
        BackgroundLearner(SlowRegressor(), maxsize=0)
    with pytest.raises(ValueError, match="sync_every must be at least 1"):
        BackgroundLearner(SlowRegressor(), sync_every=0)


def test_background_repr():
    b = BackgroundLearner(SlowRegressor(), maxsize=100)
    r = repr(b)
    assert "n_sent_=0" in r
    b.close()


def test_predict_with_t_kwarg():
    class TRegressor:
        def __init__(self):
            self.n = 0

        def learn_one(self, x, y, t=None):
            self.n += 1
            return self

        def predict_one(self, x, t=None):
            return (self.n, t)

    b = BackgroundLearner(TRegressor(), maxsize=100)
    b.learn_one({"a": 1.0}, 0.0, t=1.5)
    b.flush()
    v = b.predict_one({"a": 1.0}, t=2.0)
    assert v[1] == 2.0
    b.close()


def test_flush_on_stopped_worker():
    b = BackgroundLearner(SlowRegressor(), maxsize=100)
    b.close()
    b.flush()
    v = b.predict_one({"a": 1.0})
    assert isinstance(v, int)


def test_flush_timeout():
    b = BackgroundLearner(SlowRegressor(), maxsize=100)
    for i in range(20):
        b.learn_one({"a": float(i)}, 0.0)
    with pytest.raises(TimeoutError, match="flush timed out"):
        b.flush(timeout=0.001)
    b.flush()
    b.close()


def test_learn_one_raises_after_worker_error():
    b = BackgroundLearner(BrokenRegressor(), maxsize=100)
    b.learn_one({"a": 1.0}, 0.0)
    import time as _t

    _t.sleep(0.1)
    with pytest.raises(RuntimeError, match="boom"):
        b.learn_one({"a": 1.0}, 0.0)
    b.close()


def test_learn_one_queue_empty_race(monkeypatch):
    """Full then Empty: the sample is retried, no crash."""
    import queue as _q

    b = BackgroundLearner(SlowRegressor(), maxsize=2)
    b._queue.put_nowait(("a", 0.0, None))
    b._queue.put_nowait(("b", 0.0, None))
    orig_get = b._queue.get_nowait
    calls = [0]

    def fake_get():
        calls[0] += 1
        if calls[0] == 1:
            raise _q.Empty
        return orig_get()

    monkeypatch.setattr(b._queue, "get_nowait", fake_get)
    monkeypatch.setattr(b, "_start", lambda: None)
    b.learn_one({"a": 3.0}, 0.0)
    assert b.n_sent_ == 1
    assert b.n_dropped_ == 1
    assert b._queue.qsize() == 2


def test_learn_one_keeps_non_sample_when_full(monkeypatch):
    """Full and the oldest item is not a sample: put it back, drop the sample."""
    from dense_armor.utility.neural.background import _Flush

    b = BackgroundLearner(SlowRegressor(), maxsize=2)
    b._queue.put_nowait(_Flush())
    b._queue.put_nowait(("a", 0.0, None))
    monkeypatch.setattr(b, "_start", lambda: None)
    b.learn_one({"a": 2.0}, 0.0)
    assert b.n_sent_ == 1
    assert b.n_dropped_ == 1
    assert b._queue.qsize() == 2
