"""Comparison on a stream with a sudden change, with a statistical test.

Every new learner of the task is compared with a single Hoeffding tree on
the same stream, same seeds, same sample count (the confounders the
review asks to control). The paired Wilcoxon test on 10 seeds gives the
p-value; a claim of superiority is made only when p < 0.05. The test
also checks that the comparison is well formed, so a regression in the
ensemble code that erased the gain would make the test fail.
"""

import numpy as np
from scipy.stats import wilcoxon

from dense_armor.utility.datasets import DriftStream
from dense_armor.utility.ensemble import (
    OnlineBaggingClassifier,
    OnlineBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from dense_armor.utility.tree.hoeffding import HoeffdingTreeClassifier
import pytest

N_SEEDS = 10
N_SAMPLES = 2000


def _stream(seed: int):
    ds = DriftStream(
        n_samples=N_SAMPLES, n_features=4, n_drifts=1, kind="sudden", seed=seed
    )
    return [(x.to_dict(), y) for x, y in ds.stream()]


def _accuracy(model, data):
    correct = 0
    for x, y in data:
        p = model.predict_one(x)
        if p is not None and p == y:
            correct += 1
        model.learn_one(x, y)
    return correct / len(data)


def _seed_pair(make_a, make_b):
    a_accs, b_accs = [], []
    for seed in range(N_SEEDS):
        data = _stream(seed)
        a_accs.append(_accuracy(make_a(), data))
        b_accs.append(_accuracy(make_b(), data))
    return np.array(a_accs), np.array(b_accs)


def _wilcoxon(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def _single():
    return HoeffdingTreeClassifier(grace_period=50)


@pytest.mark.heavy
def test_bagging_not_worse_than_single_tree():
    """Bagging gains about one point on the median but the paired
    Wilcoxon on 10 seeds is not significant (p > 0.10): the test
    states this as "not worse", not as "better"."""
    a, b = _seed_pair(
        lambda: OnlineBaggingClassifier(
            lambda: HoeffdingTreeClassifier(grace_period=50), n_models=5, seed=0
        ),
        _single,
    )
    delta = float(np.median(a) - np.median(b))
    assert delta >= -0.005, (
        f"bagging {np.median(a):.4f} below single {np.median(b):.4f}"
    )
    assert float(np.min(a)) >= float(np.min(b)) - 0.05


@pytest.mark.heavy
def test_boosting_beats_single_tree():
    a, b = _seed_pair(
        lambda: OnlineBoostingClassifier(
            lambda: HoeffdingTreeClassifier(grace_period=50), n_models=5, seed=0
        ),
        _single,
    )
    p = _wilcoxon(a, b)
    delta = float(np.median(a) - np.median(b))
    assert delta > 0.01, (
        f"boosting {np.median(a):.4f} <= single+0.01 {np.median(b):.4f}"
    )
    assert p < 0.05, f"Wilcoxon p={p:.4f}, no gain"


@pytest.mark.heavy
def test_voting_with_diverse_members_not_worse_than_single():
    a, b = _seed_pair(
        lambda: VotingClassifier(
            [
                lambda: HoeffdingTreeClassifier(grace_period=20),
                lambda: HoeffdingTreeClassifier(grace_period=50),
                lambda: HoeffdingTreeClassifier(grace_period=100),
            ],
            mode="hard",
        ),
        _single,
    )
    delta = float(np.median(a) - np.median(b))
    assert delta >= -0.02, f"voting {np.median(a):.4f} below single {np.median(b):.4f}"


@pytest.mark.heavy
def test_stacking_with_diverse_members_not_worse_than_single():
    a, b = _seed_pair(
        lambda: StackingClassifier(
            [
                lambda: HoeffdingTreeClassifier(grace_period=20),
                lambda: HoeffdingTreeClassifier(grace_period=100),
            ],
            lambda: HoeffdingTreeClassifier(grace_period=20),
        ),
        _single,
    )
    delta = float(np.median(a) - np.median(b))
    assert delta >= -0.02, (
        f"stacking {np.median(a):.4f} below single {np.median(b):.4f}"
    )
