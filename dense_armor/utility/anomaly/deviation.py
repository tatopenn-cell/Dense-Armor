"""
River-compatible anomaly detectors built on Dense-Armor's streaming detectors.
Needs the optional dependency: pip install dense-armor[river]
"""
from __future__ import annotations

import numpy as np

from dense_armor.roles import AnomalyDetector

from dense_armor.utility.protect.arbiter import _robust_center_scale
from dense_armor.utility.anomaly.streaming import StreamingDeviationDetector


class StreamingDeviationScorer(AnomalyDetector):
    """`StreamingDeviationDetector` as a river anomaly detector.

    `score_one` returns the robust deviation of a value from the causal window of the values
    learned before it: `|x - median| / scale` (median/MAD, as in `arbiter.classify_segments`),
    or `|x - median|` when the window is flat. `learn_one` adds the value to the window. A score
    above `n_sigmas` is exactly the case where `StreamingDeviationDetector.update` returns True.

    Parameters
    ----------
    radius
        Half-width unit of the causal window; the window holds `radius * ref_mult` values.
    ref_mult
        Window length multiplier.
    feature
        Key of the feature to watch; `None` takes the feature with the smallest key, so the
        result does not depend on the order of the keys. River's `check_roc_auc` (a
        multi-feature fraud dataset, downloaded at test time) is skipped: this scorer watches
        one feature.
    eps
        Below this scale the window is treated as flat.

    Examples
    --------
    >>> from dense_armor.utility.anomaly.deviation import StreamingDeviationScorer
    >>> model = StreamingDeviationScorer(radius=5, ref_mult=2)
    >>> for v in [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.05]:
    ...     model.learn_one({"v": v})
    >>> model.score_one({"v": 1.0}) < 3.0
    True
    >>> model.score_one({"v": 50.0}) > 3.0
    True
    """

    def __init__(self, radius: int = 10, ref_mult: int = 3, feature: str | None = None,
                 eps: float = 1e-9):
        self.radius = radius
        self.ref_mult = ref_mult
        self.feature = feature
        self.eps = eps
        self._detector = StreamingDeviationDetector(radius=radius, ref_mult=ref_mult, eps=eps)

    def _value(self, x):
        return float(x[self.feature if self.feature is not None else min(x)])

    def score_one(self, x):
        buf = self._detector._buffer
        if len(buf) < 4:
            return 0.0
        med, scale = _robust_center_scale(np.array(buf))
        dev = abs(self._value(x) - med)
        return dev if scale < self.eps else dev / scale

    def learn_one(self, x):
        self._detector.update(self._value(x))

    def _unit_test_skips(self):
        return {"check_roc_auc"}
