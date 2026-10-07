"""Drift detector role."""

from dense_armor.roles._util import call_with_t
from dense_armor.roles.estimator import Estimator


class DriftDetector(Estimator):
    """Unsupervised drift detector over a scalar stream."""

    _supervised = False

    def __init__(self) -> None:
        self._drift_detected: bool = False
        self._warning_detected: bool = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        return self._warning_detected

    def update(self, x: float, t: float | None = None) -> 'DriftDetector':
        raise NotImplementedError

    def _reset(self) -> None:
        self._drift_detected = False
        self._warning_detected = False

    def update_many(self, X, t=None) -> 'DriftDetector':
        for x in X:
            call_with_t(self.update, x, t=t)
        return self
