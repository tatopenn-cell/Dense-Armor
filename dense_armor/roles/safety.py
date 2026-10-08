"""Safety as a first-class role.

Two things make an estimator safe inside a control loop:

- a **health state** the outer loop can poll: ``warming_up``,
  ``nominal``, ``drifting``, ``degraded``. The state comes from the
  estimator's own residuals: a running statistic tells when the model
  starts to disagree with the real robot.
- a **guard**: an anomaly scorer with a threshold, a drift detector, or
  a set of physical limits. When the guard fires, the model does not
  learn from the sample (so a bad reading cannot poison the state), and
  the output is a fallback value.

Checkpoints are versioned and checksummed. A corrupted file, or one
written with a different format version, is refused with a clear error,
so a control loop never silently restores a bad state.

References
----------
Page, E. S. (1954). Continuous inspection schemes. Biometrika 41,
    100-114.
"""
import hashlib
import pickle
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from dense_armor.roles import Root, Estimator


CHECKPOINT_VERSION = 1


class Health(str, Enum):
    """Health states of an estimator in a control loop."""

    WARMING_UP = "warming_up"
    NOMINAL = "nominal"
    DRIFTING = "drifting"
    DEGRADED = "degraded"


class HealthMonitor(Root):
    """Running residual tracker that maps to a :class:`Health` state.

    Keeps the last ``window`` absolute residuals in a deque. The state
    is derived from the recent mean vs a baseline (Page 1954):

    - while fewer than ``warmup`` residuals have been seen:
      ``warming_up``;
    - if the recent mean is within ``drift_mult`` times the baseline:
      ``nominal``;
    - if it is beyond ``drift_mult``: ``drifting``;
    - if it is beyond ``degraded_mult``: ``degraded``.

    Args:
        window: size of the recent-residual window.
        warmup: samples before the state can leave ``warming_up``.
        drift_mult: multiplier of the baseline above which the state is
            ``drifting``.
        degraded_mult: multiplier above which the state is ``degraded``.
    """

    def __init__(
        self,
        window: int = 100,
        warmup: int = 20,
        drift_mult: float = 2.0,
        degraded_mult: float = 5.0,
    ) -> None:
        self.window = window
        self.warmup = warmup
        self.drift_mult = drift_mult
        self.degraded_mult = degraded_mult
        self.recent_: list[float] = []
        self.baseline_: Optional[float] = None
        self.n_ = 0

    def _state_from_window(self) -> Health:
        if self.n_ <= self.warmup:
            return Health.WARMING_UP
        baseline = self.baseline_
        if baseline is None or not self.recent_:
            return Health.NOMINAL
        recent_mean = sum(self.recent_) / len(self.recent_)
        if recent_mean > self.degraded_mult * baseline:
            return Health.DEGRADED
        if recent_mean > self.drift_mult * baseline:
            return Health.DRIFTING
        return Health.NOMINAL

    def update(self, residual: float) -> Health:
        """Feed one residual and return the current :class:`Health`."""
        self.n_ += 1
        self.recent_.append(abs(residual))
        if len(self.recent_) > self.window:
            self.recent_.pop(0)
        state = self._state_from_window()
        if (
            state is Health.NOMINAL
            and self.baseline_ is None
            and self.n_ > self.warmup
        ):
            self.baseline_ = sum(self.recent_) / len(self.recent_)
        return state

    @property
    def state(self) -> Health:
        """Current state without updating the window."""
        return self._state_from_window()


class SafeEstimator(Estimator):
    """Wrap an estimator with a guard and a fallback.

    Every sample is scored first. If the guard flags it, the wrapped
    model does not learn from the sample and the output is the fallback
    value.

    The wrapped ``model`` can be any estimator: a regressor, a
    classifier, a transformer, an anomaly detector. Only the methods
    the model actually exposes are called; the rest pass through as
    ``None``.

    Args:
        model: the estimator to wrap.
        guard: callable ``x -> bool`` returning ``True`` for a flagged
            sample. ``None`` disables guarding.
        fallback: value returned while flagged. ``None`` means "hold the
            last good prediction".
        monitor: optional :class:`HealthMonitor`; a default one is
            created when not given.

    Examples:
        >>> from dense_armor.roles.safety import SafeEstimator
        >>> from dense_armor.utility.stats.moments import RunningMoments
        >>> guard = lambda s: s["x"] > 1e6
        >>> safe = SafeEstimator(RunningMoments(), guard=guard, fallback=0.0)
        >>> _ = safe.learn_one({"x": 1.0}, 0.0)
        >>> safe.is_flagged({"x": 1e7})
        True
        >>> safe.health.value
        'warming_up'
    """

    def __init__(
        self,
        model: Any,
        guard: Optional[Callable[[Any], bool]] = None,
        fallback: Any = None,
        monitor: Optional[HealthMonitor] = None,
    ) -> None:
        self.model = model
        self.guard = guard
        self.fallback = fallback
        self.monitor = monitor if monitor is not None else HealthMonitor()
        self._last_good_: Any = fallback
        self.n_blocked_ = 0

    def is_flagged(self, x: Any) -> bool:
        """Return ``True`` if the guard flags ``x``."""
        if self.guard is None:
            return False
        return bool(self.guard(x))

    @property
    def health(self) -> Health:
        """Current :class:`Health` state."""
        return self.monitor.state

    def learn_one(self, x: Any, y: Any = None, t: Optional[float] = None):
        """Learn from ``(x, y)`` unless the guard flags ``x``."""
        self._time_step(t)
        if self.is_flagged(x):
            self.n_blocked_ += 1
            return self
        m = self.model
        if hasattr(m, "predict_one") and y is not None:
            yhat = m.predict_one(x, t=t)
            if yhat is not None:
                try:
                    self.monitor.update(abs(float(y) - float(yhat)))
                except (TypeError, ValueError):
                    pass
        if hasattr(m, "learn_one"):
            try:
                m.learn_one(x, y, t=t)
            except TypeError:
                try:
                    m.learn_one(x, y)
                except TypeError:
                    m.learn_one(x)
        return self

    def predict_one(
        self,
        x: Any,
        t: Optional[float] = None,
        return_std: bool = False,
        return_estimate: bool = False,
    ) -> Any:
        """Predict with the wrapped model, or the fallback if flagged."""
        self._time_step(t)
        if self.is_flagged(x):
            self.n_blocked_ += 1
            if self.fallback is not None:
                return self.fallback
            return self._last_good_
        m = self.model
        if not hasattr(m, "predict_one"):
            return None
        out = m.predict_one(x, t=t)
        self._last_good_ = out
        return out

    def score_one(self, x: Any, t: Optional[float] = None) -> float:
        """Delegate ``score_one`` to the wrapped model, or NaN if absent."""
        self._time_step(t)
        m = self.model
        if not hasattr(m, "score_one"):
            return float("nan")
        return float(m.score_one(x, t=t))

    def save(self, path: str | Path) -> Path:
        """Save the learned state with format version and SHA-256.

        Args:
            path: destination file.

        Returns:
            The path written.
        """
        state = {
            "version": CHECKPOINT_VERSION,
            "model_module": type(self.model).__module__,
            "model_class": type(self.model).__name__,
            "timestamp": time.time(),
            "n_blocked": self.n_blocked_,
            "state_dict": self.model.state_dict()
            if hasattr(self.model, "state_dict")
            else {},
        }
        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(blob).hexdigest()
        payload = {"sha256": digest, "blob": blob}
        p = Path(path)
        p.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        return p

    def restore(self, path: str | Path) -> "SafeEstimator":
        """Restore the learned state from a saved checkpoint.

        Args:
            path: source file.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: if the checkpoint format version does not match,
                or the checksum does not match.
            TypeError: if the stored model class does not match.
        """
        p = Path(path)
        payload = pickle.loads(p.read_bytes())
        blob = payload["blob"]
        digest = hashlib.sha256(blob).hexdigest()
        if digest != payload["sha256"]:
            raise ValueError(f"checkpoint {p} is corrupted: SHA-256 mismatch")
        state = pickle.loads(blob)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"checkpoint version {state.get('version')} "
                f"does not match current {CHECKPOINT_VERSION}"
            )
        if state.get("model_class") != type(self.model).__name__:
            raise TypeError(
                f"checkpoint model class {state.get('model_class')} "
                f"does not match current {type(self.model).__name__}"
            )
        if hasattr(self.model, "load_state_dict"):
            self.model.load_state_dict(state["state_dict"])
        self.n_blocked_ = int(state.get("n_blocked", 0))
        return self
