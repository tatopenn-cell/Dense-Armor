"""Seeds of ``DriftStream`` for which the change really bites.

The change of a ``DriftStream`` gives the clusters of the new concept a new random direction, and on about
half of the seeds a classifier fitted before the change stays right after it (the labels alternate, but the
new direction can put the old rule back on the same side). The benchmarks and the figures of the forests use
the first seeds for which a Hoeffding tree that learns until the change and then stops has an accuracy
below ``MAX_ACCURACY`` on the samples after the change. The rule is fixed here, before the numbers of the
forests are looked at.
"""

import numpy as np

from dense_armor.core.online.tree import HoeffdingTreeClassifier
from dense_armor.core.signals.datasets import DriftStream

N, D = 2000, 4
CUT = N // 2
MAX_ACCURACY = 0.5


def frozen_accuracy(X, y) -> float:
    """Accuracy after the change of a tree that learns until the change and then stops."""
    names = [f"x{i}" for i in range(X.shape[1])]
    tree = HoeffdingTreeClassifier(grace_period=50)
    hits = []
    for i, (xi, yi) in enumerate(zip(X, y)):
        d = dict(zip(names, map(float, xi)))
        if i < CUT:
            tree.learn_one(d, int(yi))
        else:
            hits.append(float(tree.predict_one(d) == yi))
    return float(np.mean(hits))


def drift_seeds(n: int = 5) -> list:
    """The first ``n`` seeds whose change makes the frozen tree wrong (accuracy below ``MAX_ACCURACY``)."""
    seeds = []
    seed = 0
    while len(seeds) < n:
        X, y = DriftStream(
            n_samples=N, n_features=D, n_drifts=1, kind="sudden", seed=seed
        ).data()
        if frozen_accuracy(X, y) < MAX_ACCURACY:
            seeds.append(seed)
        seed += 1
    return seeds


if __name__ == "__main__":
    print(drift_seeds())
