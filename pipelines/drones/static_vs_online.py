"""Static vs online: a CART tree and an online Hoeffding tree on the same stream.

The same fault stream, one batch CART tree and one online Hoeffding
tree, twenty seeds. Both see the same 16 features and predict the same
label (the fault flag). The question is: how much accuracy does the
offline model buy, and at what cost in time and memory?

The batch model is :class:`BatchDecisionTreeClassifier`
(``dense_armor/utility/tree/batch.py``), a CART tree fitted on the whole
training set in one call. The online model is
:class:`HoeffdingTreeClassifier`
(``dense_armor/utility/tree/hoeffding.py``), which learns one sample at
a time and predicts each test sample before learning from it
(prequential).

Training and test are a 50/50 shuffled split of the stream. The fault
starts at 60% of the stream, so the post-fault part is 40% of the
samples; both halves contain both classes. The batch fits on the
training half; the online starts empty and walks the test half in the
shuffled order. Both are scored on the same test half, same samples,
same order.

The 16 features per sample: gyro x, y, z (3), acc x, y, z (3), gps_v
north, east, down (3), motor 1-4 (4), battery_v, current_a, payload_kg
(3), for a total of 16.

Reference for CART: Breiman, Friedman, Olshen, Stone (1984),
*Classification and Regression Trees*, Wadsworth, chapter 2. The book
is not on arXiv and was not read for this page; the gini impurity
``I(t) = 1 - sum_k p_k^2`` and the split rule (minimise the weighted
impurity of the two children) are the classical ones, and they are also
stated by Bourel et al. 2026 (arXiv:2606.13984, page 8, page 14,
equation for the Gini index).

Example:
    >>> from pipelines.drones.static_vs_online import run
    >>> r = run(n_seeds=2, n_samples=600)
    >>> r.n_seeds
    2
    >>> len(r.accuracy_batch)
    2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.tree.batch import BatchDecisionTreeClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
from pipelines.drones.io import synthetic_flight

N_FEATURES = 16
DEFAULT_MAX_DEPTH = 6


@dataclass
class StaticVsOnlineResult:
    """Result of the static-vs-online comparison.

    Attributes:
        n_seeds: number of seeds run.
        n_samples: samples per stream.
        warmup: warmup samples, the size of the training half.
        accuracy_batch: per-seed accuracy of the batch model (n_seeds,).
        accuracy_online: per-seed accuracy of the online model (n_seeds,).
        median_accuracy_batch: median of accuracy_batch.
        median_accuracy_online: median of accuracy_online.
        p_two_sided: paired Wilcoxon two-sided p-value (batch vs online).
        a12: A12 effect size (probability that a random batch value is
            larger than a random online value).
        a12_ci95: bootstrap 95% interval of A12.
        seeds_batch_better: number of seeds where batch > online.
        seeds_online_better: number of seeds where online > batch.
        seeds_tie: number of seeds where they are equal.
        time_per_sample_batch_ms: median per-sample time of the batch
            model (fit on the training half plus predict on the test).
        time_per_sample_online_ms: median per-sample time of the online
            model (predict plus learn on the test half).
        memory_batch_kb: peak memory of the batch model, in KB.
        memory_online_kb: peak memory of the online model, in KB.
        batch_name: name of the batch class.
        online_name: name of the online class.
    """

    n_seeds: int
    n_samples: int
    warmup: int
    accuracy_batch: np.ndarray
    accuracy_online: np.ndarray
    median_accuracy_batch: float
    median_accuracy_online: float
    p_two_sided: float
    a12: float
    a12_ci95: tuple[float, float]
    seeds_batch_better: int
    seeds_online_better: int
    seeds_tie: int
    time_per_sample_batch_ms: float
    time_per_sample_online_ms: float
    memory_batch_kb: float
    memory_online_kb: float
    batch_name: str
    online_name: str


def _features(sample: dict) -> dict:
    """The 16-feature dict of one sample."""
    g = np.asarray(sample.get("gyro", [0.0, 0.0, 0.0]), dtype=float)
    a = np.asarray(sample.get("acc", [0.0, 0.0, 0.0]), dtype=float)
    v = np.asarray(sample.get("gps_v", [0.0, 0.0, 0.0]), dtype=float)
    m = np.asarray(sample.get("motor", [0.0, 0.0, 0.0, 0.0]), dtype=float)
    return {
        "gyro_x": float(g[0]), "gyro_y": float(g[1]), "gyro_z": float(g[2]),
        "acc_x": float(a[0]), "acc_y": float(a[1]), "acc_z": float(a[2]),
        "gps_n": float(v[0]), "gps_e": float(v[1]), "gps_d": float(v[2]),
        "motor_1": float(m[0]), "motor_2": float(m[1]),
        "motor_3": float(m[2]), "motor_4": float(m[3]),
        "battery_v": float(sample.get("battery_v", 0.0)),
        "current_a": float(sample.get("current_a", 0.0)),
        "payload_kg": float(sample.get("payload_kg", 0.0)),
    }


def _a12(a: np.ndarray, b: np.ndarray) -> float:
    """A12 effect size: P(a > b) + 0.5 * P(a == b)."""
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _a12_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 2000,
) -> tuple[float, float]:
    """Bootstrap 95% interval of A12."""
    k = len(a)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, size=k)
        vals[i] = _a12(a[idx], b[idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Paired two-sided Wilcoxon p-value, or 1.0 on constant differences."""
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def run(
    n_seeds: int = 20,
    n_samples: int = 2000,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> StaticVsOnlineResult:
    """Run the static-vs-online comparison.

    Args:
        n_seeds: number of seeds. Default 20.
        n_samples: samples per stream. Default 2000.
        max_depth: maximum depth of the CART tree. Default 6.

    Returns:
        :class:`StaticVsOnlineResult`.
    """
    import os
    import time

    import psutil

    acc_batch = np.zeros(n_seeds)
    acc_online = np.zeros(n_seeds)
    times_batch = np.zeros(n_seeds)
    times_online = np.zeros(n_seeds)
    mem_batch = np.zeros(n_seeds)
    mem_online = np.zeros(n_seeds)

    for seed in range(n_seeds):
        stream = list(synthetic_flight(
            n=n_samples, seed=seed, fault="actuator_loss"))
        X = [_features(s) for s in stream]
        y = [int(s["fault_started"]) for s in stream]

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_samples)
        n_train = n_samples // 2
        tr_idx = perm[:n_train]
        te_idx = perm[n_train:]
        X_tr = [X[i] for i in tr_idx]
        y_tr = [y[i] for i in tr_idx]
        X_te = [X[i] for i in te_idx]
        y_te = [y[i] for i in te_idx]

        # batch CART
        rss = psutil.Process(os.getpid()).memory_info().rss
        t0 = time.perf_counter()
        clf_b = BatchDecisionTreeClassifier(max_depth=max_depth).fit(X_tr, y_tr)
        preds_b = [clf_b.predict_one(x) for x in X_te]
        t_b = (time.perf_counter() - t0) / (n_train + len(X_te))
        mem_b = psutil.Process(os.getpid()).memory_info().rss - rss
        acc_batch[seed] = float(np.mean(
            [p == t for p, t in zip(preds_b, y_te)]))
        times_batch[seed] = t_b * 1000.0
        mem_batch[seed] = mem_b / 1024.0

        # online Hoeffding
        rss = psutil.Process(os.getpid()).memory_info().rss
        t0 = time.perf_counter()
        clf_o = HoeffdingTreeClassifier(grace_period=50)
        ok = 0
        for x, t in zip(X_te, y_te):
            p = clf_o.predict_one(x)
            if p is not None and p == t:
                ok += 1
            clf_o.learn_one(x, t)
        t_o = (time.perf_counter() - t0) / len(X_te)
        mem_o = psutil.Process(os.getpid()).memory_info().rss - rss
        acc_online[seed] = ok / len(X_te)
        times_online[seed] = t_o * 1000.0
        mem_online[seed] = mem_o / 1024.0

    p = _wilcoxon(acc_batch, acc_online)
    a12 = _a12(acc_batch, acc_online)
    ci_lo, ci_hi = _a12_ci(
        acc_batch, acc_online, np.random.default_rng(0))

    return StaticVsOnlineResult(
        n_seeds=n_seeds,
        n_samples=n_samples,
        warmup=n_samples // 2,
        accuracy_batch=acc_batch,
        accuracy_online=acc_online,
        median_accuracy_batch=float(np.median(acc_batch)),
        median_accuracy_online=float(np.median(acc_online)),
        p_two_sided=p,
        a12=a12,
        a12_ci95=(ci_lo, ci_hi),
        seeds_batch_better=int((acc_batch > acc_online).sum()),
        seeds_online_better=int((acc_online > acc_batch).sum()),
        seeds_tie=int((acc_batch == acc_online).sum()),
        time_per_sample_batch_ms=float(np.median(times_batch)),
        time_per_sample_online_ms=float(np.median(times_online)),
        memory_batch_kb=float(np.median(mem_batch)),
        memory_online_kb=float(np.median(mem_online)),
        batch_name="BatchDecisionTreeClassifier (CART)",
        online_name="HoeffdingTreeClassifier",
    )
