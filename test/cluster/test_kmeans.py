import numpy as np

from dense_armor.checks import check_estimator
from dense_armor.utility.cluster.kmeans import OnlineKMeans


def test_kmeans_three_blobs_ari():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
    X_list, y_list = [], []
    for i, c in enumerate(centers):
        for _ in range(200):
            X_list.append(rng.normal(c, 0.5))
            y_list.append(i)
    X = np.stack(X_list)
    y_true = np.asarray(y_list)
    perm = rng.permutation(len(X))
    X, y_true = X[perm], y_true[perm]

    km = OnlineKMeans(k=3, seed=0)
    for v in X:
        km.learn_one({"x": v.tolist()})
    y_pred = np.array([km.predict_one({"x": v.tolist()}) for v in X])
    ari = adjusted_rand_score(y_true, y_pred)

    sk = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    ari_sk = adjusted_rand_score(y_true, sk.labels_)

    print(f"ARI OnlineKMeans = {ari:.4f}")
    print(f"ARI sklearn KMeans = {ari_sk:.4f}")
    assert ari >= 0.9, f"ARI too low: {ari:.4f}"
    assert ari >= ari_sk - 0.1, f"OnlineKMeans {ari:.4f} vs sklearn {ari_sk:.4f}"


def test_kmeans_halflife_follows_moving_blobs():
    rng = np.random.default_rng(2)
    km = OnlineKMeans(k=3, halflife=30.0, seed=0)
    starts = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    ends = np.array([[0.0, 4.0], [8.0, 4.0], [0.0, 12.0]])
    for _ in range(300):
        for c in starts:
            km.learn_one({"x": rng.normal(c, 0.2).tolist()})
    for _ in range(1000):
        for c in ends:
            km.learn_one({"x": rng.normal(c, 0.2).tolist()})
    dists = []
    for c in ends:
        d = min(float(np.linalg.norm(km.centers_[j] - c)) for j in range(3))
        dists.append(d)
    print(f"halflife final distances to true centres: {[round(d, 4) for d in dists]}")
    assert max(dists) < 0.6, f"centres did not follow: {dists}"


def test_kmeans_predict_and_transform():
    km = OnlineKMeans(k=2, warmup=4)
    for v in [0.0, 0.1, 10.0, 10.1, 0.2, 9.9]:
        km.learn_one({"x": [v]})
    assert km.predict_one({"x": [0.0]}) != km.predict_one({"x": [10.0]})
    assert sorted(round(float(c[0]), 1) for c in km.centers_) == [0.1, 10.0]
    out = km.transform_one({"x": [0.0]})
    assert "cluster" in out and "distances" in out
    assert len(out["distances"]) == 2


def test_kmeans_nan_counted():
    km = OnlineKMeans(k=2)
    km.learn_one({"x": [0.0]})
    km.learn_one({"x": [float("nan")]})
    km.learn_one({"x": [1.0]})
    assert km.n_missing == 1


def test_check_estimator_kmeans():
    check_estimator(OnlineKMeans(k=3))
