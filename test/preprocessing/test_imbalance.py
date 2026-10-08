import random

import pytest

from dense_armor.utility.evaluate import progressive_val_score
from dense_armor.utility.metrics import Recall
from dense_armor.utility.preprocessing.imbalance import QueueResampler


def _make_classifier():
    module = pytest.importorskip(
        "dense_armor.utility.learn.online_classifiers"
    )
    for name in ("OnlineGaussianNB", "OnlineSoftmaxRegression"):
        cls = getattr(module, name, None)
        if cls is not None:
            return cls()
    pytest.skip("no online classifier available")


def _separable_stream(n: int = 5000, p_pos: float = 0.01) -> list:
    rng = random.Random(0)
    stream = []
    for _ in range(n):
        y = 1 if rng.random() < p_pos else 0
        m = 2.0 if y == 1 else -2.0
        x0 = rng.gauss(m, 1.0)
        x1 = rng.gauss(m, 1.0)
        stream.append(({"x0": x0, "x1": x1}, y))
    return stream


def test_queue_resampler_balances_queues():
    qr = QueueResampler(_make_classifier(), queue_size=10)
    for i in range(100):
        y = 1 if i % 100 == 0 else 0
        qr.learn_one({"x0": float(i), "x1": float(i)}, y=y)
    assert qr.n_seen_ == 100
    assert len(qr.q_positive_) == 1
    assert len(qr.q_negative_) == 10


def test_queue_resampler_boosts_minority_recall():
    stream = _separable_stream(n=5000, p_pos=0.01)
    base = progressive_val_score(
        stream, _make_classifier(), Recall(positive=1)
    )
    qr = progressive_val_score(
        stream,
        QueueResampler(_make_classifier(), queue_size=25),
        Recall(positive=1),
    )
    base_recall = float(base.get())
    qr_recall = float(qr.get())
    print(f"recall without QueueResampler: {base_recall:.4f}")
    print(f"recall with    QueueResampler: {qr_recall:.4f}")
    print(f"stream length: {len(stream)}")
    assert qr_recall > base_recall


def test_check_estimator_queue():
    from dense_armor.checks import check_estimator

    class _Stub:
        def learn_one(self, x, y, t=None):
            return self

        def predict_one(self, x, t=None):
            return 0

        def predict_proba_one(self, x, t=None):
            return {0: 1.0}

    from dense_armor.roles import Classifier

    class _StubClf(Classifier):
        def learn_one(self, x, y, t=None):
            return self

        def predict_one(self, x, t=None):
            return 0

        def predict_proba_one(self, x, t=None):
            return {0: 1.0}

    check_estimator(QueueResampler(_StubClf(), queue_size=5))
