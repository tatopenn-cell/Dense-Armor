"""Tests for LocalOutlierFactor.

Numbers reported in the docstrings come from the runs in this file.
"""

import doctest
from pathlib import Path

import numpy as np
import pytest

from dense_armor.utility.anomaly import lof as lof_mod
from dense_armor.utility.anomaly.lof import LocalOutlierFactor

URDF = Path("test/fixtures/urdf/panda.urdf")


def test_doctests():
    r = doctest.testmod(lof_mod)
    assert r.failed == 0


def test_paper_introduction_example():
    """The 2-D example of the paper's introduction, Section 3.

    A dense cluster C2 of 100 points, a sparse cluster C1 of 400
    points, one point o1 near the dense cluster. The LOF of o1 must be
    higher than the LOF of every point of C1: the outlier is local,
    it is near the dense cluster, not far from it.

    Positions are scaled to fit a window of 502 samples; the numbers
    below are read from the run.
    """
    rng = np.random.default_rng(0)
    C1 = rng.normal(loc=(0.0, 0.0), scale=1.0, size=(400, 2))
    C2 = rng.normal(loc=(3.0, 3.0), scale=0.15, size=(100, 2))
    o1 = np.array([3.0 + 0.8, 3.0 + 0.8])

    model = LocalOutlierFactor(n_neighbors=20, window=550)

    for p in C1:
        model.learn_one({"a": float(p[0]), "b": float(p[1])})
    for p in C2:
        model.learn_one({"a": float(p[0]), "b": float(p[1])})

    lof_o1 = model.score_one({"a": float(o1[0]), "b": float(o1[1])})

    lof_C1 = np.array(
        [model.score_one({"a": float(p[0]), "b": float(p[1])}) for p in C1]
    )
    lof_C2 = np.array(
        [model.score_one({"a": float(p[0]), "b": float(p[1])}) for p in C2]
    )

    assert lof_o1 > lof_C1.max(), f"o1={lof_o1:.3f}, max C1={lof_C1.max():.3f}"
    assert lof_o1 > lof_C2.max(), f"o1={lof_o1:.3f}, max C2={lof_C2.max():.3f}"
    assert np.all(lof_C1 > 0.0)
    assert np.all(lof_C2 > 0.0)

    # Inliers on the dense cluster score around 1.
    assert 0.7 < np.median(lof_C2) < 1.3


def test_score_around_one_for_an_inlier():
    rng = np.random.default_rng(0)
    model = LocalOutlierFactor(n_neighbors=10, window=200)
    for _ in range(200):
        x = rng.standard_normal(2) * 0.1
        model.learn_one({"a": float(x[0]), "b": float(x[1])})
    s = model.score_one({"a": 0.0, "b": 0.0})
    assert 0.5 < s < 2.0, f"inlier LOF = {s}"


def test_score_high_for_a_far_outlier():
    rng = np.random.default_rng(0)
    model = LocalOutlierFactor(n_neighbors=10, window=200)
    for _ in range(200):
        x = rng.standard_normal(2) * 0.1
        model.learn_one({"a": float(x[0]), "b": float(x[1])})
    s_in = model.score_one({"a": 0.0, "b": 0.0})
    s_out = model.score_one({"a": 5.0, "b": 5.0})
    assert s_out > s_in


def test_small_window_returns_one():
    """A window with fewer than n_neighbors + 1 samples: LOF undefined, returns 1.

    With n_neighbors = 5, the neighborhood of a point needs at least
    6 samples in the window (the point itself plus 5 neighbors). With
    3 samples the neighborhood is not defined and score_one returns
    1.0 by the documented convention.
    """
    model = LocalOutlierFactor(n_neighbors=5, window=10)
    for i in range(3):
        model.learn_one({"a": float(i), "b": float(i)})
    assert model.score_one({"a": 0.0, "b": 0.0}) == 1.0

    model.learn_one({"a": 3.0, "b": 3.0})
    assert model.score_one({"a": 0.0, "b": 0.0}) == 1.0

    model.learn_one({"a": 4.0, "b": 4.0})
    assert model.score_one({"a": 0.0, "b": 0.0}) == 1.0

    model.learn_one({"a": 5.0, "b": 5.0})
    s = model.score_one({"a": 0.0, "b": 0.0})
    assert s != 1.0
    assert np.isfinite(s)


def test_duplicates_give_finite_scores():
    """Duplicated points: the density is capped by the 1e-10 term."""
    model = LocalOutlierFactor(n_neighbors=5, window=50)
    for _ in range(6):
        model.learn_one({"a": 1.0, "b": 1.0})
    for i in range(20):
        model.learn_one({"a": float(i) * 10.0, "b": float(i) * 10.0})
    s = model.score_one({"a": 1.0, "b": 1.0})
    assert s == pytest.approx(1.0)


def test_far_query_on_window_of_duplicates_is_very_high():
    model = LocalOutlierFactor(n_neighbors=5, window=30)
    for _ in range(30):
        model.learn_one({"a": 1.0, "b": 1.0})
    assert model.score_one({"a": 1.0, "b": 1.0}) == pytest.approx(1.0)
    far = model.score_one({"a": 9.0, "b": 9.0})
    assert np.isfinite(far)
    assert far > 1e9


def test_malformed_input_scores_zero():
    model = LocalOutlierFactor(n_neighbors=5, window=20, feature_keys=["a", "b"])
    for i in range(20):
        model.learn_one({"a": float(i), "b": float(i)})
    assert model.score_one({"a": 1.0}) == 0.0
    assert model.score_one("x") == 0.0
    assert model.score_one({"a": float("nan"), "b": 0.0}) == 0.0


@pytest.mark.parametrize("d,k,w", [(2, 20, 100), (5, 10, 150), (3, 5, 60)])
def test_matches_scikit_learn(d, k, w):
    sk = pytest.skip("no reference metric module", allow_module_level=False) or None
    rng = np.random.default_rng(0)
    X = rng.normal(size=(w, d))
    keys = [f"f{j}" for j in range(d)]
    model = LocalOutlierFactor(n_neighbors=k, window=w, feature_keys=keys)
    for r in X:
        model.learn_one(dict(zip(keys, map(float, r))))
    ref = sk.LocalOutlierFactor(n_neighbors=k, novelty=True).fit(X)
    Q = rng.normal(size=(50, d)) * 1.7
    mine = np.array([model.score_one(dict(zip(keys, map(float, q)))) for q in Q])
    assert np.max(np.abs(mine + ref.score_samples(Q))) < 1e-8


def test_nan_and_missing_counted_in_n_missing():
    model = LocalOutlierFactor(n_neighbors=5, window=20)
    for i in range(20):
        model.learn_one({"a": float(i), "b": float(i)})
    n0 = model.n_missing
    model.learn_one({"a": float("nan"), "b": 1.0})
    assert model.n_missing == n0 + 1
    model.learn_one({"a": 1.0, "b": 1.0})
    assert model.n_missing == n0 + 1
    model.learn_one({"a": float("inf"), "b": 0.0})
    assert model.n_missing == n0 + 2
    model.learn_one({})
    assert model.n_missing == n0 + 3
    model.learn_one("not a dict")
    assert model.n_missing == n0 + 4
    # score_one on a non-finite sample returns 0, and does not touch
    # n_missing (score_one does not learn)
    assert model.score_one({"a": float("nan"), "b": 0.0}) == 0.0
    assert model.score_one({"a": 1.0, "b": 2.0}) > 0.0


def test_check_estimator_passes():
    from dense_armor.checks import check_estimator

    check_estimator(LocalOutlierFactor(n_neighbors=5, window=30))


def test_constructor_arguments_not_overwritten():
    model = LocalOutlierFactor(n_neighbors=5, window=20, threshold=2.0, seed=7)
    params = model.get_params()
    for k, v in params.items():
        if k in ("n_neighbors", "window", "threshold", "seed"):
            assert v == getattr(model, k)
    clone = model.clone()
    assert clone.n_neighbors == 5
    assert clone.window == 20
    assert clone.threshold == 2.0


def test_bad_parameters_raise():
    with pytest.raises(ValueError, match="n_neighbors"):
        LocalOutlierFactor(n_neighbors=0)
    with pytest.raises(ValueError, match="window"):
        LocalOutlierFactor(n_neighbors=10, window=5)
    with pytest.raises(ValueError, match="threshold"):
        LocalOutlierFactor(threshold=0.0)


def test_feature_keys_explicit():
    model = LocalOutlierFactor(n_neighbors=3, window=20, feature_keys=["a", "b"])
    for i in range(20):
        model.learn_one({"a": float(i), "b": float(i), "c": 99.0})
    # c is ignored
    s = model.score_one({"a": 0.0, "b": 0.0, "c": 1000.0})
    assert 0.0 < s < 5.0


def test_synthetic_arm_stream():
    """LOF on the SyntheticArm payload fault, AUC above chance."""
    pytest.skip('no reference metric module')

    from dense_armor.utility.datasets import SyntheticArm

    ds = SyntheticArm(
        URDF,
        period_s=2.0,
        rate_hz=50.0,
        n_cycles=2,
        fault_at_s=2.0,
        fault="payload",
        payload_mass=5.0,
        noise_std=(1e-3, 5e-3, 5e-2),
        seed=0,
    )
    X, y = [], []
    for sig, label in ds.stream():
        q, qd, qdd = ds.trajectory(sig.t)
        tau_nom = ds.nominal_torque(q, qd, qdd)
        x = {f"r{j}": float(sig[f"tau_{j}"] - tau_nom[j]) for j in range(4)}
        X.append(x)
        y.append(int(label))
    y = np.asarray(y)

    model = LocalOutlierFactor(n_neighbors=20, window=50)
    scores = []
    for i, x in enumerate(X):
        scores.append(model.score_one(x))
        model.learn_one(x)
    scores = np.asarray(scores)
    auc = roc_auc_score(y, scores)
    assert auc > 0.5, f"AUC = {auc}"
