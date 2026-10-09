"""Paths of Protected with score-only and drift detectors, and unsupervised models."""

from dense_armor.roles.protection import Protected


class _Mean:
    def __init__(self):
        self.n = 0

    def learn_one(self, x):
        self.n += 1

    def score_one(self, x):
        return float(self.n)


class _ScoreOnly:
    def score_one(self, x):
        return 1.0


class _Drift:
    def __init__(self):
        self.drift_detected = False

    def update(self, v):
        self.drift_detected = v > 5.0


def test_protected_score_only_detector_never_flags():
    p = Protected(model=_Mean(), detector=_ScoreOnly())
    p.learn_one({"v": 1.0})
    assert p.model.n == 1 and p.score_one({"v": 1.0}) == 1.0


def test_protected_drift_detector_flags_and_scores_inf():
    p = Protected(model=_Mean(), detector=_Drift())
    p.learn_one({"v": 9.0})
    assert p.model.n == 0
    assert p.score_one({"v": 9.0}) == float("inf")
