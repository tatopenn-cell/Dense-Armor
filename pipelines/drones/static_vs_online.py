"""Static vs online: a batch bagging and an online bagging on the same stream.

Both models are bagging of ``HoeffdingTreeClassifier`` with the same
number of trees (5) and the same tree hyperparameters. They differ in
how the trees see the data:

- **batch**: for each tree, draw a bootstrap sample of the warm-up
  (same size, with replacement), and fit the tree on the whole sample
  at once via ``tree.fit(X_boot, y_boot)``. Predict by majority vote.
- **online**: :class:`OnlineBaggingClassifier` of the library, which
  gives each tree a Poisson(1) weight per sample and learns one sample
  at a time (Oza and Russell 2001, section 2).

The two protocols are the subject of Oza and Russell 2001: in the
batch, each tree sees a bootstrap of size N; in the online bagging,
each tree sees each sample k times with k ~ Poisson(1). In the limit
N -> infinity the counts have the same distribution, so the two
learners build the same trees in expectation. This page measures the
difference on a finite stream of 2000 samples.

Training and test are a **50/50 shuffled split** of the stream. The
fault is at 60% of the stream, so both halves contain both classes
(about 40% healthy, 60% fault). The batch model fits on the training
half and predicts the test half. The online model starts empty, walks
the test half in the same shuffled order, predicts each sample and
learns from it (prequential). Both see the same test samples in the
same order; the only difference is that the batch model saw 1000
labelled samples before the test started, and the online model did
not.

This is the comparison the page is about: how much does 1000 labelled
samples bought offline buy you, in accuracy, time and memory, over
learning only online? A warm-up of the first N samples would not work:
with the fault starting at 60% of the stream, the first N samples are
all healthy, so the batch model would learn "always 0" and the
comparison would be against a trivial baseline, not against the
online model.

The 16 features per sample are the same as the other pipelines of the
section: gyro x, y, z; acc x, y, z; gps_v north, east, down; motor 1-4;
battery_v; current_a; payload_kg. The label is fault_started (0 normal,
1 post-fault). The stream is actuator_loss, 2000 samples, 60% at the
fault.

No external library is used. ``OnlineBaggingClassifier``,
``HoeffdingTreeClassifier``, ``fit_batch`` and ``Root.fit`` are the
library's own (``fit_batch`` was added to ``bagging.py`` for this
page and is the batch protocol of Oza and Russell 2001).

Example:
    >>> from pipelines.drones.static_vs_online import run
    >>> r = run(n_seeds=2, n_samples=600, n_models=3)
    >>> r.n_seeds
    2
    >>> len(r.accuracy_batch)
    2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dense_armor.utility.ensemble.bagging import OnlineBaggingClassifier
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
from pipelines.drones.io import synthetic_flight

N_FEATURES = 16
N_MODELS = 5
GRACE_PERIOD = 50


@dataclass
class StaticVsOnlineResult:
    """Result of the static-vs-online comparison.

    Attributes:
        n_seeds: number of seeds run.
        n_samples: samples per stream.
        warmup: warm-up samples, used for fitting the batch model and
            for the first learning pass of the online model.
        accuracy_batch: per-seed accuracy of the batch model on the
            test part (n_seeds,).
        accuracy_online: per-seed accuracy of the online model on the
            test part (n_seeds,).
        median_accuracy_batch: median of accuracy_batch.
        median_accuracy_online: median of accuracy_online.
        time_per_sample_batch_ms: median per-sample time of the batch
            model, fit plus predict divided by the samples it sees.
        time_per_sample_online_ms: median per-sample time of the online
            model, learn plus predict on the test part.
        memory_batch_kb: peak memory of the batch model, in KB.
        memory_online_kb: peak memory of the online model, in KB.
        batch_name: name of the batch protocol.
        online_name: name of the online class.
    """

    n_seeds: int
    n_samples: int
    warmup: int
    accuracy_batch: np.ndarray
    accuracy_online: np.ndarray
    median_accuracy_batch: float
    median_accuracy_online: float
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



def run(
    n_seeds: int = 20,
    n_samples: int = 2000,
    fault_at_fraction: float = 0.6,
    n_models: int = 5,
) -> StaticVsOnlineResult:
    """Run the static-vs-online comparison, 20 seeds.

    Training and test are a 50/50 shuffled split of the stream. The
    batch model fits on the training half with ``fit_batch`` (each
    member sees a bootstrap of the training half). The online model
    starts empty, walks the test half in the same shuffled order,
    predicts each sample before learning from it. Both are scored on
    the same test half.

    Args:
        n_seeds: number of seeds. Default 20.
        n_samples: samples per stream. Default 2000.
        fault_at_fraction: fraction of the stream at which the fault
            starts. Default 0.6.
        n_models: number of trees in both models. Default 5.

    Returns:
        :class:`StaticVsOnlineResult`.
    """
    import time
    import tracemalloc

    acc_batch = np.zeros(n_seeds)
    acc_online = np.zeros(n_seeds)
    times_batch = np.zeros(n_seeds)
    times_online = np.zeros(n_seeds)
    mem_batch = np.zeros(n_seeds)
    mem_online = np.zeros(n_seeds)

    def _tree_factory():
        return HoeffdingTreeClassifier(grace_period=GRACE_PERIOD)

    for seed in range(n_seeds):
        stream = list(synthetic_flight(n=n_samples, seed=seed, fault="actuator_loss"))
        X = [_features(s) for s in stream]
        y = [int(s["fault_started"]) for s in stream]

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_samples)
        n_train = n_samples // 2
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        y_te = [y[i] for i in test_idx]

        # batch
        tracemalloc.start()
        t0 = time.perf_counter()
        clf_b = OnlineBaggingClassifier(
            _tree_factory, n_models=n_models, seed=seed,
        )
        clf_b.fit_batch(X_tr, y_tr, seed=seed)
        preds_b = [int(clf_b.predict_one(x) or 0) for x in X_te]
        t_batch = (time.perf_counter() - t0) / (n_train + len(X_te))
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        acc_batch[seed] = float(
            np.mean([p == t for p, t in zip(preds_b, y_te)])
        )
        times_batch[seed] = t_batch * 1000.0
        mem_batch[seed] = peak_b / 1024.0

        # online
        tracemalloc.start()
        t0 = time.perf_counter()
        clf_o = OnlineBaggingClassifier(
            _tree_factory, n_models=n_models, seed=seed,
        )
        ok = 0
        for x, yv in zip(X_te, y_te):
            p = clf_o.predict_one(x)
            if p is not None and p == yv:
                ok += 1
            clf_o.learn_one(x, yv)
        t_online = (time.perf_counter() - t0) / len(X_te)
        _, peak_o = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        acc_online[seed] = ok / len(X_te)
        times_online[seed] = t_online * 1000.0
        mem_online[seed] = peak_o / 1024.0

    return StaticVsOnlineResult(
        n_seeds=n_seeds,
        n_samples=n_samples,
        warmup=n_samples // 2,
        accuracy_batch=acc_batch,
        accuracy_online=acc_online,
        median_accuracy_batch=float(np.median(acc_batch)),
        median_accuracy_online=float(np.median(acc_online)),
        time_per_sample_batch_ms=float(np.median(times_batch)),
        time_per_sample_online_ms=float(np.median(times_online)),
        memory_batch_kb=float(np.median(mem_batch)),
        memory_online_kb=float(np.median(mem_online)),
        batch_name="OnlineBaggingClassifier.fit_batch (bootstrap)",
        online_name="OnlineBaggingClassifier.learn_one (Poisson(1))",
    )
