"""
Anomaly detectors compatible with an existing online-learning library, built on Dense-Armor's streaming detectors.
Needs the optional dependency: the streaming extra
"""

import numpy as np

from dense_armor.roles import AnomalyDetector
from dense_armor.utility.anomaly.streaming import (
    StreamingDeviationDetector,
    _feature_value,
)
from dense_armor.utility.protect.arbiter import _robust_center_scale


class StreamingDeviationScorer(AnomalyDetector):
    """`StreamingDeviationDetector` as an anomaly detector of an existing online-learning library.

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
        result does not depend on the order of the keys. The `check_roc_auc` of that library (a
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

    n_missing_ = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _value(self, x):
        return _feature_value(x, self.feature)

    def score_one(self, x):
        buf = self._detector._buffer
        v = self._value(x)
        if v is None or len(buf) < 4:
            return 0.0
        med, scale = _robust_center_scale(np.array(buf))
        dev = abs(v - med)
        return dev if scale < self.eps else dev / scale

    def learn_one(self, x):
        v = self._value(x)
        if v is None:
            self.n_missing_ += 1
            return
        self._detector.update(v)

    def _unit_test_skips(self):
        return {"check_roc_auc"}
