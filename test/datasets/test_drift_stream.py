"""Tests for the drift stream."""

import numpy as np
import pytest

from dense_armor.utility.datasets import DriftStream


def test_drift_stream_shapes():
    ds = DriftStream(n_samples=200, n_features=3, n_drifts=2, kind="sudden", seed=0)
    X, y = ds.data()
    assert X.shape == (200, 3)
    assert y.shape == (200,)
    assert set(np.unique(y)) <= {0, 1}
    assert ds.drift_positions.shape == (2,)
    assert ds.drift_positions[0] == pytest.approx(1 / 3)
    assert ds.drift_positions[1] == pytest.approx(2 / 3)


def test_drift_stream_rejects_bad_kind():
    with pytest.raises(ValueError):
        DriftStream(100, 2, 1, "random")


def test_drift_stream_rejects_bad_counts():
    with pytest.raises(ValueError):
        DriftStream(0, 2, 1, "sudden")
    with pytest.raises(ValueError):
        DriftStream(100, 0, 1, "sudden")
    with pytest.raises(ValueError):
        DriftStream(100, 2, -1, "sudden")


def test_drift_stream_no_drift_is_stationary():
    ds = DriftStream(n_samples=500, n_features=2, n_drifts=0, kind="sudden", seed=1)
    X, y = ds.data()
    assert ds.drift_positions.shape == (0,)
    assert X.shape == (500, 2)
    assert len(np.unique(y)) == 2


def test_drift_stream_concepts_public():
    ds = DriftStream(n_samples=100, n_features=2, n_drifts=3, kind="gradual", seed=2)
    _ = ds.data()
    assert ds.concepts_.shape == (100,)
    assert ds.concepts_.max() <= 3
    assert ds.concepts_.min() >= 0


def test_drift_stream_stream_yields_signals():
    ds = DriftStream(n_samples=30, n_features=2, n_drifts=1, kind="sudden", seed=0)
    out = list(ds.stream())
    assert len(out) == 30
    sig0, y0 = out[0]
    assert sig0.names == ["x0", "x1"]
    assert sig0.t == 0.0
    assert y0 in (0, 1)


def test_drift_stream_online_nb_drops_after_drift():
    from dense_armor.utility.evaluate import progressive_val_score
    from dense_armor.utility.learn.online_classifiers import (
        OnlineGaussianNB,
    )
    from dense_armor.utility.metrics import Accuracy
    from dense_armor.utility.metrics.rolling import Rolling

    ds = DriftStream(
        n_samples=2000,
        n_features=2,
        n_drifts=1,
        kind="sudden",
        spacing=999.0,
        seed=0,
    )
    stream = ds.stream()
    clf = OnlineGaussianNB()
    metric = Rolling(Accuracy(), window=20)
    result = progressive_val_score(stream, clf, metric, every=5)
    assert isinstance(result, tuple)
    _, trace = result
    pre = float(np.mean(trace[40:160]))
    post_min = float(np.min(trace[200:240]))
    assert pre > post_min + 0.15


def test_gradual_transition_is_continuous_across_the_drift():
    ds = DriftStream(
        n_samples=20000, n_features=2, n_drifts=1, kind="gradual", spacing=20.0, seed=0
    )
    ds.data()
    c = ds.concepts_
    before = c[9000:10000].mean()
    after = c[10000:11000].mean()
    assert 0.3 < before < 0.5
    assert 0.5 < after < 0.7
