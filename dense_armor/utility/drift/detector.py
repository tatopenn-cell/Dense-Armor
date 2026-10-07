
"""River-compatible CUSUM drift detector built on Dense-Armor's CUSUM implementation."""
from __future__ import annotations
from collections import deque
import numpy as np

from dense_armor.roles import DriftDetector

from dense_armor.utility.protect.arbiter import _robust_center_scale
from dense_armor.utility.drift.cusum import one_sided_arl, two_sided_arl


class CUSUMDriftDetector(DriftDetector):
    """CUSUM drift detector for river, matching the batch `cusum_detector` behaviour.

    Parameters
    ----------
    radius, ref_mult
        Causal window span = radius * ref_mult, same convention as `cusum_detector`.
    k
        CUSUM slack, in robust-sigma units.
    h
        Decision threshold, in accumulated robust-sigma units.
    two_sided
        If True, both upward and downward shifts are monitored. If False, only
        upward shifts are monitored.
    eps
        Degenerate-scale guard, same convention as `cusum_detector`.
    reference
        "adaptive" recomputes median and scale from a sliding causal window every
        step, matching `cusum_detector(reference="adaptive")`. The reference drifts
        with the data, so a sustained shift is caught at its leading edge, then
        stops accumulating once the window catches up. The detection floor for a
        sustained step is roughly 2.2 robust sigmas at h=20.
        "fixed" computes median and scale once from the first `span` values and
        never updates, matching `cusum_detector(reference="fixed")` and Page's
        original scheme. Detects smaller sustained shifts at the cost of a higher
        false-alarm rate on stationary data.

    Examples
    --------
    >>> from dense_armor.utility.drift.detector import CUSUMDriftDetector
    >>> import numpy as np
    >>> det = CUSUMDriftDetector(reference="fixed")
    >>> rng = np.random.default_rng(0)
    >>> stream = list(rng.normal(0, 1, 500)) + list(rng.normal(1, 1, 500))
    >>> for i, x in enumerate(stream):
    ...     _ = det.update(x)
    ...     if det.drift_detected:
    ...         print(i)
    ...         break
    516
    """

    def __init__(self, radius: int = 10, ref_mult: int = 3, k: float = 0.5,
                 h: float = 20.0, two_sided: bool = True, eps: float = 1e-9,
                 reference: str = "adaptive"):
        super().__init__()
        if reference not in ("adaptive", "fixed"):
            raise ValueError(
                f"reference must be 'adaptive' or 'fixed', got {reference!r}"
            )
        self.radius = radius
        self.ref_mult = ref_mult
        self.k = k
        self.h = h
        self.two_sided = two_sided
        self.eps = eps
        self.reference = reference
        self._span = radius * ref_mult
        self._buffer = deque(maxlen=self._span)
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._seen = 0
        self._fixed_ready = False
        self._fixed_med = None
        self._fixed_scale = None

    def _reset(self):
        super()._reset()
        self._buffer.clear()
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._seen = 0
        self._fixed_ready = False
        self._fixed_med = None
        self._fixed_scale = None

    def _unit_test_skips(self):
        return set()

    def _accumulate(self, z: float) -> None:
        self._s_pos = max(0.0, self._s_pos + z - self.k)
        if self.two_sided:
            self._s_neg = min(0.0, self._s_neg + z + self.k)
        if self._s_pos > self.h:
            self._drift_detected = True
            self._s_pos = 0.0
        elif self.two_sided and self._s_neg < -self.h:
            self._drift_detected = True
            self._s_neg = 0.0

    def update(self, x: float):
        self._drift_detected = False

        if self.reference == "fixed":
            self._seen += 1
            if not self._fixed_ready:
                self._buffer.append(x)
                if self._seen == self._span:
                    arr = np.array(self._buffer)
                    if np.all(np.isfinite(arr)):
                        med, scale = _robust_center_scale(arr)
                        if np.isfinite(scale) and scale >= self.eps:
                            self._fixed_med = med
                            self._fixed_scale = scale
                    self._fixed_ready = True
                return self
            if self._fixed_med is None:
                return self
            if not np.isfinite(x):
                return self
            z = (x - self._fixed_med) / self._fixed_scale
            self._accumulate(z)
            return self

        if not np.isfinite(x):
            self._buffer.append(x)
            return self
        if len(self._buffer) < 4:
            self._buffer.append(x)
            return self
        arr = np.array(self._buffer)
        med, scale = _robust_center_scale(arr)
        if not np.isfinite(scale) or scale < self.eps:
            self._buffer.append(x)
            return self
        z = (x - med) / scale
        self._accumulate(z)
        self._buffer.append(x)
        return self

    def expected_false_alarm_run(self) -> float:
        """Expected number of samples between false alarms under no change."""
        return two_sided_arl(0.0, self.k, self.h)

    def expected_detection_delay(self, shift: float) -> float:
        """Expected number of samples to detect a standardized mean shift `shift`."""
        return two_sided_arl(shift, self.k, self.h)
