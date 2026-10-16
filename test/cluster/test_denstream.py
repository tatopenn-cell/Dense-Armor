"""Tests for DenStream.

Reference
---------
Cao, F., Ester, M., Qian, W., Zhou, A. (2006). "Density-Based Clustering
over an Evolving Data Stream with Noise." In Proceedings of the 2006 SIAM
International Conference on Data Mining (SDM), pp. 328-339.

Formulas used in the tests
--------------------------
- Fading function, paper Section 3: a point with time stamp ``T``
  contributes weight ``f(t - T) = 2 ** (-lambda * (t - T))`` at current
  time ``t``, with ``lambda > 0``.
- Pruning rule, paper Section 4.1: a p-micro-cluster whose weight
  ``w`` drops below ``beta * mu`` is deleted (its memory is released
  for new clusters).
- Consequence: a cluster with weight ``w_before`` at time ``t0`` has
  weight ``w_before * 2 ** (-lambda * (t - t0))`` at time ``t`` if no
  new point is merged. It falls below ``beta * mu`` after
  ``dt = log2(w_before / (beta * mu)) / lambda`` time steps. This is
  the formula used in :func:`test_denstream_blob_fades`.
- The paper also defines (eq. 4.1) the pruning-check interval
  ``T_p = (1 / lambda) * log(beta * mu / (beta * mu - 1))``. This is
  the interval between pruning checks, not the fade time of a single
  cluster. The tests use the fade time, not ``T_p``.

The test parameters follow the paper's experiments (page 8): ``beta``
and ``mu`` are chosen so that ``beta * mu > 1``, which is required for
the pruning interval ``T_p`` to be defined.
"""

import math

import numpy as np
import pytest

from dense_armor.checks import check_estimator
from dense_armor.utility.cluster.denstream import DenStream


def _blobs_with_noise(rng, n_per=150, noise=100):
    centers = np.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 8.0]])
    X_list, y_list = [], []
    for i, c in enumerate(centers):
        for _ in range(n_per):
            X_list.append(rng.normal(c, 0.3))
            y_list.append(i)
    for _ in range(noise):
        X_list.append(rng.uniform(-15.0, 15.0, size=2))
        y_list.append(-1)
    X = np.stack(X_list)
    y_true = np.asarray(y_list)
    perm = rng.permutation(len(X))
    return X[perm], y_true[perm]


def _assign(X, clusters, threshold):
    y = np.full(len(X), -1, dtype=int)
    for i, v in enumerate(X):
        best_d, best_k = float("inf"), -1
        for k, cl in enumerate(clusters):
            d = float(np.linalg.norm(v - np.array(cl["center"])))
            if d < best_d:
                best_d, best_k = d, k
        if best_d <= threshold:
            y[i] = best_k
    return y


def test_denstream_blobs_plus_noise():
    """Three well-separated blobs plus uniform noise: DenStream
    recovers the three blob centres, assigns most blob points, and
    rejects most noise. It may form small extra clusters from noise;
    the test does not require that it finds exactly three."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    X, y_true = _blobs_with_noise(rng)
    ds = DenStream(eps=1.5, beta=0.3, mu=10.0, decay=0.001)
    for i, v in enumerate(X):
        ds.learn_one({"x": v.tolist()}, t=float(i))
    clusters, _ = ds.macro_clusters()

    # Verify the three expected blob centres are recovered. DenStream
    # may also produce small extra clusters from noise: with mu=10 and
    # beta=0.3 the noise threshold is beta*mu/4 = 0.75, so a handful of
    # noise points may survive as a cluster. The test checks that the
    # three true blobs are present, not that no noise cluster forms.
    expected_centres = np.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 8.0]])
    recovered = np.array([c["center"] for c in clusters])
    for e in expected_centres:
        d = np.linalg.norm(recovered - e, axis=1).min()
        assert d < 2.0, (
            f"blob at {e} not recovered; nearest centre "
            f"is {d:.2f} away. All centres: {recovered.tolist()}"
        )

    y_pred = _assign(X, clusters, threshold=3.0)
    n_noise = int((y_true == -1).sum())
    n_blob = int((y_true != -1).sum())

    n_noise_assigned = int(((y_true == -1) & (y_pred != -1)).sum())
    n_blob_assigned = int(((y_true != -1) & (y_pred != -1)).sum())

    assert n_blob_assigned >= 0.9 * n_blob, (
        f"only {n_blob_assigned}/{n_blob} blob points assigned"
    )
    ari = adjusted_rand_score(y_true[y_true != -1], y_pred[y_true != -1])
    assert ari >= 0.7, f"ARI on blob points = {ari:.4f}"

    # A density-based algorithm with mu=10 absorbs some noise that
    # falls within eps of a blob: this is expected (Cao et al. 2006,
    # Section 5 discusses cluster overlap). The uniform noise is in
    # [-15, 15]^2 (area 900); the union of the three r=3 discs around
    # the blob centres covers ~85 area, so ~9% of the noise is
    # geometric neighbours. The 30% threshold allows for the seed
    # specific sample and stays well above the geometric expectation.
    assert n_noise_assigned <= 0.3 * n_noise, (
        f"{n_noise_assigned}/{n_noise} noise points were assigned"
    )


def test_denstream_blob_fades():
    """A p-micro-cluster whose weight drops below ``beta * mu`` is
    deleted (paper Section 4.1). With only far-away data arriving, the
    old cluster fades at ``log2(w / (beta * mu)) / decay`` time steps,
    derived from the fading function ``f(t) = 2 ** (-decay * t)`` of
    paper Section 3.

    Parameters use ``mu = 10`` so that ``beta * mu = 2 > 1``, matching
    the regime the paper assumes (page 8).
    """
    rng = np.random.default_rng(1)
    ds = DenStream(eps=1.0, beta=0.2, mu=10.0, decay=0.2)
    for i in range(200):
        v = rng.normal(0.0, 0.2, size=2)
        ds.learn_one({"x": v.tolist()}, t=float(i))

    clusters_before, _ = ds.macro_clusters()
    near_before = [
        c for c in clusters_before if np.linalg.norm(np.array(c["center"])) < 3.0
    ]
    assert near_before, "no cluster formed around (0, 0)"

    w_before = sum(c["weight"] for c in near_before)
    assert w_before > 0.0

    # From the fading function f(t) = 2 ** (-decay * t) and the pruning
    # rule w < beta * mu, a cluster with weight w_before at time 0
    # falls below beta * mu after
    #     w_before * 2 ** (-decay * dt) = beta * mu
    # -> dt = log2(w_before / (beta * mu)) / decay.
    dt = math.log2(w_before / (ds.beta * ds.mu)) / ds.decay
    assert dt > 0, "fade time must be positive"

    start = 200
    for i in range(start, start + math.ceil(dt) + 50):
        v = rng.normal(50.0, 0.2, size=2)
        ds.learn_one({"x": v.tolist()}, t=float(i))

    clusters_after, _ = ds.macro_clusters()
    still = [c for c in clusters_after if np.linalg.norm(np.array(c["center"])) < 5.0]
    assert not still, (
        f"old cluster should have faded after ~{dt:.1f} steps, but {len(still)} remain"
    )


def test_denstream_isolated_point_stays_outlier():
    """An isolated point far from any existing cluster must not create
    a new core cluster (paper Algorithm 1, step 3: it becomes an
    o-micro-cluster, not a p-micro-cluster)."""
    ds = DenStream(eps=0.5, beta=0.4, mu=1.0, decay=0.1)
    for i in range(30):
        ds.learn_one({"x": [0.0, 0.0]}, t=float(i))
    clusters_before, _ = ds.macro_clusters()
    n_before = len(clusters_before)

    ds.learn_one({"x": [100.0, 100.0]}, t=30.0)

    clusters_after, _ = ds.macro_clusters()
    assert len(clusters_after) == n_before, (
        "an isolated point created a new core cluster"
    )


def test_denstream_nan_counted_and_skipped():
    """A NaN sample increments ``n_missing`` and does not update
    clusters."""
    ds = DenStream()
    ds.learn_one({"x": [0.0, 0.0]}, t=0.0)
    clusters_before, _ = ds.macro_clusters()
    n_before = len(clusters_before)

    ds.learn_one({"x": [float("nan"), 0.0]}, t=1.0)

    assert ds.n_missing == 1
    clusters_after, _ = ds.macro_clusters()
    assert len(clusters_after) == n_before, "a NaN sample changed the cluster state"


def test_denstream_boundary_assign_threshold():
    """The assign threshold used in :func:`test_denstream_blobs_plus_noise`
    (3.0) separates in from out: a point at distance 2.0 from the
    cluster's center is assigned, a point at distance 4.0 is not."""
    rng = np.random.default_rng(2)
    ds = DenStream(eps=2.0, beta=0.3, mu=10.0, decay=0.01)
    for i in range(200):
        v = rng.normal(0.0, 0.1, size=2)
        ds.learn_one({"x": v.tolist()}, t=float(i))

    clusters, _ = ds.macro_clusters()
    assert clusters, "no cluster formed"

    center = np.array(clusters[0]["center"])
    inside_pt = (center + np.array([2.0, 0.0])).reshape(1, 2)
    outside_pt = (center + np.array([4.0, 0.0])).reshape(1, 2)
    inside = _assign(inside_pt, clusters, threshold=3.0)
    outside = _assign(outside_pt, clusters, threshold=3.0)
    assert inside[0] != -1, (
        f"point at distance ~2.0 from center was rejected: "
        f"center={center}, point={inside_pt[0]}"
    )
    assert outside[0] == -1, (
        f"point at distance ~4.0 from center was assigned: "
        f"center={center}, point={outside_pt[0]}"
    )


def test_check_estimator_denstream():
    """A fresh DenStream passes every applicable check."""
    check_estimator(DenStream())


def test_check_estimator_rejects_broken_denstream():
    """``check_estimator`` must reject an estimator that violates a
    contract, not only pass on good ones. A ``clone`` that returns the
    original object instead of a copy violates ``check_clone`` and
    ``check_clone_independent``; ``check_estimator`` must catch it.
    """

    class BrokenClone(DenStream):
        def clone(self, new_params=None, include_attributes=False):
            return self

    with pytest.raises(AssertionError):
        check_estimator(BrokenClone())

