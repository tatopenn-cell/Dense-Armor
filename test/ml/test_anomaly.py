"""Hand cases of the anomaly detection of the machine learning package."""

import doctest
import math

import numpy as np
import pytest

import dense_armor.ml.anomaly as A
from dense_armor.ml.anomaly import (
    ChauvenetScorer,
    HampelFilter,
    LocalOutlierFactor,
    SigmaClipScorer,
    TukeyScorer,
)

SPIKE = [1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0]


def test_doctests():
    r = doctest.testmod(A)
    assert r.failed == 0 and r.attempted >= 8


def test_hampel_hand_case():
    m = HampelFilter(radius=2).fit(SPIKE)
    assert m.anomalies_ == [4] and m.cleaned_.tolist() == [1.0] * 9
    x = [0.0, 1.0, -1.0, 0.0, 1.0, 5.0, 0.0, 1.0, -1.0]
    m = HampelFilter(radius=4, n_sigmas=3.0).fit(x)
    assert m.anomalies_ == [5]
    assert HampelFilter(radius=1).fit([1.0, 2.0]).anomalies_ == []


def test_tukey_hand_case():
    m = TukeyScorer(radius=2).fit(SPIKE)
    assert m.anomalies_ == [4] and m.cleaned_[4] == 1.0
    assert TukeyScorer(radius=1).fit([1.0, 2.0, 3.0]).anomalies_ == []
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert TukeyScorer(radius=4).fit(x + [100.0]).anomalies_ == [5]


def test_chauvenet_hand_case():
    m = ChauvenetScorer(radius=4).fit([0.0, 0.0, 0.0, 0.0, 8.0])
    assert m.anomalies_ == [4]
    d = 2.0
    assert abs(5 * math.erfc(d / math.sqrt(2)) - 0.2275) < 1e-3
    assert ChauvenetScorer(radius=4).fit([3.0] * 6).anomalies_ == []
    assert ChauvenetScorer(radius=1).fit([1.0, 2.0]).anomalies_ == []


def test_sigma_clip_hand_case():
    x = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 50.0]
    m = SigmaClipScorer(radius=4).fit(x)
    assert m.anomalies_ == [8] and m.cleaned_[8] == 2.0
    m = SigmaClipScorer(radius=2).fit(SPIKE)
    assert m.anomalies_ == [4]
    assert SigmaClipScorer(radius=1).fit([1.0, 2.0]).anomalies_ == []


def test_sigma_clip_removes_the_outliers_of_the_window():
    x = [20.0] + [1.0] * 9 + [100.0] + [1.0] * 10
    m = SigmaClipScorer(radius=20).fit(x)
    assert m.anomalies_ == [0, 10]
    assert m.cleaned_[0] == 1.0 and m.cleaned_[10] == 1.0
    plain = SigmaClipScorer(radius=20, max_iters=0).fit(x)
    assert plain.anomalies_ == [10]


def test_sigma_clip_empty_kept_set():
    m = SigmaClipScorer(radius=4, n_sigmas=0.5).fit([1.0, -1.0, 1.0, -1.0, 1.0])
    assert m.anomalies_ == []


def test_lof_hand_case():
    X = [[0.0], [1.0], [2.0], [3.0], [10.0]]
    m = LocalOutlierFactor(n_neighbors=2).fit(X)
    assert m.kdist_.tolist() == [2.0, 1.0, 1.0, 2.0, 8.0]
    assert np.allclose(m.lrd_[:4], 2 / 3) and abs(m.lrd_[4] - 2 / 15) < 1e-8
    assert np.allclose(m.lof_, [1, 1, 1, 1, 5], atol=1e-6)
    s = m.score([[1.5], [30.0]])
    assert s[1] > 5 > s[0] > 0.5
    with pytest.raises(ValueError):
        LocalOutlierFactor(n_neighbors=5).fit(X)


def test_lof_duplicates_finite():
    m = LocalOutlierFactor(n_neighbors=2).fit([[1.0]] * 5)
    assert np.isfinite(m.lof_).all() and np.allclose(m.lof_, 1.0)


def test_stopped():
    assert len(A.STOPPED) == 9 and all(A.STOPPED.values())
