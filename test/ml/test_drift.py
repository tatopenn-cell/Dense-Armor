"""Hand cases of the change detection of the machine learning package."""

import doctest

import numpy as np

import dense_armor.ml.drift as D
from dense_armor.ml.drift import CUSUMDriftDetector


def test_doctests():
    r = doctest.testmod(D)
    assert r.failed == 0 and r.attempted >= 3


def test_hand_case():
    x = [0.0, 1.0, -1.0, 0.0, 1.0, -1.0] * 5 + [6.0] * 5
    m = CUSUMDriftDetector(radius=3, ref_mult=2, k=0.5, h=5.0).fit(x)
    s = 1.4826
    assert m.median_ == 0.0 and abs(m.scale_ - s) < 1e-12
    assert abs(m.cusum_[30] - (6.0 / s - 0.5)) < 1e-12
    assert np.flatnonzero(m.flagged_).tolist() == [31, 33]


def test_downward_and_skips():
    x = [0.0, 1.0, -1.0, 0.0, 1.0, -1.0] + [-6.0, np.nan, -6.0]
    m = CUSUMDriftDetector(radius=3, ref_mult=2, k=0.5, h=5.0).fit(x)
    assert np.flatnonzero(m.flagged_).tolist() == [8] and m.cusum_[7] == 0.0
    assert m.cusum_[8] < -5.0


def test_degenerate():
    m = CUSUMDriftDetector(radius=2, ref_mult=2).fit([1.0] * 10)
    assert not m.flagged_.any()
    m = CUSUMDriftDetector(radius=5, ref_mult=5).fit([1.0, 2.0])
    assert not m.flagged_.any() and np.isnan(m.median_)
    m = CUSUMDriftDetector(radius=2, ref_mult=1).fit([1.0, np.nan, 3.0, 4.0])
    assert not m.flagged_.any()


def test_stopped():
    assert len(D.STOPPED) == 4 and all(D.STOPPED.values())
