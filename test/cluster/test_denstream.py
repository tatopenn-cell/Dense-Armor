import math

import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.cluster.denstream import DenStream


def _blobs_with_noise(rng, n_per=150, noise=25):
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


def test_denstream_blobs_plus_noise():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    X, y_true = _blobs_with_noise(rng)
    ds = DenStream(eps=1.5, beta=0.3, mu=2.0, decay=0.01)
    for i, v in enumerate(X):
        ds.learn_one({"x": v.tolist()}, t=float(i))
    clusters, _ = ds.macro_clusters()
    print(f"n macro-clusters = {len(clusters)}")
    print(f"n pMC kept = {len(ds.pmc_)}")
    print(f"n oMC kept = {len(ds.omc_)}")
    assert 2 <= len(clusters) <= 4, f"n_clusters = {len(clusters)}"

    y_pred = np.full(len(X), -1, dtype=int)
    for i, v in enumerate(X):
        best_d, best_k = float("inf"), -1
        for k, cl in enumerate(clusters):
            d = float(np.linalg.norm(v - np.array(cl["center"])))
            if d < best_d:
                best_d, best_k = d, k
        if best_d <= 3.0:
            y_pred[i] = best_k
    mask = (y_true != -1) & (y_pred != -1)
    ari = adjusted_rand_score(y_true[mask], y_pred[mask])
    noise_assigned = int(((y_true == -1) & (y_pred != -1)).sum())
    print(f"ARI on non-noise assigned = {ari:.4f}")
    print(f"noise points assigned = {noise_assigned}")
    assert ari >= 0.7, f"ARI = {ari:.4f}"


def test_denstream_blob_fades():
    rng = np.random.default_rng(1)
    ds = DenStream(eps=1.0, beta=0.2, mu=2.0, decay=0.2)
    for i in range(200):
        v = rng.normal(0.0, 0.2, size=2)
        ds.learn_one({"x": v.tolist()}, t=float(i))
    w_before = sum(mc.w for mc in ds.pmc_ if np.linalg.norm(mc.c) < 3.0)
    assert w_before > 0.0, "no cluster formed around (0, 0)"
    dt = math.log2(w_before / (ds.beta * ds.mu)) / ds.decay
    print(f"w_before = {w_before:.3f}")
    print(f"predicted fade dt = {dt:.2f} steps")
    start = 200
    for i in range(start, start + int(dt) + 100):
        v = rng.normal(50.0, 0.2, size=2)
        ds.learn_one({"x": v.tolist()}, t=float(i))
    still = [mc for mc in ds.pmc_ if np.linalg.norm(mc.c) < 5.0]
    print(f"old pMCs still present = {len(still)}")
    assert not still, "old blob should have faded away"


def test_denstream_isolated_point_stays_outlier():
    ds = DenStream(eps=0.5, beta=0.4, mu=1.0, decay=0.1)
    for i in range(30):
        ds.learn_one({"x": [0.0, 0.0]}, t=float(i))
    n_pmc_before = len(ds.pmc_)
    ds.learn_one({"x": [100.0, 100.0]}, t=30.0)
    assert len(ds.pmc_) == n_pmc_before
    assert any(np.linalg.norm(mc.c - np.array([100.0, 100.0])) < 1e-6 for mc in ds.omc_)


def test_denstream_nan_counted():
    ds = DenStream()
    ds.learn_one({"x": [0.0, 0.0]}, t=0.0)
    ds.learn_one({"x": [float("nan"), 0.0]}, t=1.0)
    assert ds.n_missing == 1


def test_check_estimator_denstream():
    check_estimator(DenStream())
