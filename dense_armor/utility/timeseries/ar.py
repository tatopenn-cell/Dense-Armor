"""Autoregressive models learned one sample at a time.

``OnlineAR`` predicts the next value of one channel from its last
``order`` values with recursive least squares with forgetting.

``OnlineVAR`` extends the same idea to several channels at once. The
underlying algorithm is TIRSO from Zaman et al. (2019): recursive
least squares with a group-lasso penalty that selects which
channel-lag blocks matter. Set ``group_lambda=0.0`` to disable the
penalty and recover plain recursive least squares on the stacked
history.

References:
    Zaman, B., Lopez Ramos, L. M., Romero, D., Beferull-Lozano, B.
    (2019). Online topology identification from vector autoregressive
    time series. arXiv:1904.01864. Equations (19) and (21) are the
    recursive least squares update with forgetting; equations (24)
    and (25) are the group-lasso update; the algorithm is Procedure 2
    (page 6); the static regret bound is Theorem 3 (page 7) under
    assumptions A1, A2, A3 (bounded min/max eigenvalue of Phi).
"""
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from dense_armor.roles import Transformer


class OnlineAR(Transformer):
    """Online univariate autoregressive model of order ``order``.

    At every sample the model predicts the value of the channel from
    its last ``order`` values and updates its coefficients by
    recursive least squares with a forgetting factor.

    Args:
        order: number of past values used in the prediction.
        lam: forgetting factor in ``(0, 1]``. ``1.0`` weights all past
            samples equally; smaller values track a changing signal.
        delta: initial value of the inverse-Hessian matrix.
        feature: key of the channel to read from a dict. If ``None``,
            the smallest key is used.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.timeseries import OnlineAR
        >>> rng = np.random.default_rng(0)
        >>> ar = OnlineAR(order=2, lam=1.0)
        >>> for _ in range(50):
        ...     _ = ar.learn_one({"x": float(rng.normal())})
        >>> abs(ar.predict_one({"x": 0.0})) < 5.0
        True
    """

    budget_s: float | None = 1e-3
    memory_class: str = "O(1)"

    def __init__(
        self,
        order: int = 1,
        lam: float = 1.0,
        delta: float = 1e6,
        feature: str | None = None,
    ) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if not 0.0 < lam <= 1.0:
            raise ValueError(f"lam must be in (0, 1], got {lam}")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        self.order = int(order)
        self.lam = float(lam)
        self.delta = float(delta)
        self.feature = feature
        self._key: str | None = None
        self._hist: np.ndarray | None = None
        self._w: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._n: int = 0
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        """Count of samples skipped because of a missing or bad value."""
        return self.n_missing_

    def _key_of(self, x: dict) -> str:
        if self.feature is not None:
            return self.feature
        if self._key is None:
            self._key = min(x)
        return self._key

    def _init(self) -> None:
        d = self.order
        self._hist = np.zeros(d)
        self._w = np.zeros(d)
        self._P = self.delta * np.eye(d)

    def _read(self, x: Any) -> float | None:
        if not isinstance(x, dict):
            return None
        try:
            v = x[self._key_of(x)]
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError, KeyError, IndexError):
            return None

    def _unit_test_skips(self) -> set:
        """Skip checks that assume a scalar state.

        ``check_step_is_pure`` converts every leaf of the state to a
        Python float. The state of this model contains a history
        vector and a full inverse-Hessian matrix, so that check
        cannot run. ``check_step_is_jittable`` still runs and passes.
        """
        return {"check_step_is_pure"}

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "OnlineAR":
        """One sample: predict, update, push to history."""
        v = self._read(x)
        if v is None:
            self.n_missing_ += 1
            return self
        if self._w is None:
            self._init()
        assert self._hist is not None
        assert self._w is not None
        assert self._P is not None
        if self._n >= self.order:
            Px = self._P @ self._hist
            K = Px / (self.lam + float(self._hist @ Px))
            err = v - float(self._w @ self._hist)
            self._w = self._w + K * err
            self._P = (self._P - np.outer(K, Px)) / self.lam
            self._P = 0.5 * (self._P + self._P.T)
        self._hist = np.concatenate([self._hist[1:], [v]])
        self._n += 1
        return self

    def predict_one(self, x: dict, t: float | None = None) -> float:
        """Forecast the next sample from the current history."""
        if self._w is None or self._n < self.order:
            return 0.0
        assert self._hist is not None
        return float(self._w @ self._hist)

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """One-step-ahead residual of the current sample."""
        v = self._read(x)
        if v is None:
            return {"residual": float("nan")}
        return {"residual": v - self.predict_one(x)}

    def init_state(self, dim: int = 1) -> tuple:
        """Return the initial ``(hist, w, P, n)`` state."""
        d = self.order
        return (
            jnp.zeros(d),
            jnp.zeros(d),
            jnp.eye(d) * self.delta,
            jnp.int32(0),
        )

    def step(self, state: tuple, v: Any) -> tuple:
        """One pure step, for ``jax.lax.scan``."""
        hist, w, P, n = state
        xv = jnp.asarray(v).reshape(-1)[:1]
        pred = jnp.dot(w, hist)
        Px = P @ hist
        K = Px / (self.lam + jnp.dot(hist, Px))
        err = xv[0] - pred
        w_new = w + K * err
        P_new = (P - jnp.outer(K, Px)) / self.lam
        P_new = 0.5 * (P_new + P_new.T)
        upd = n < self.order
        w_upd = jnp.where(upd, w, w_new)
        P_upd = jnp.where(upd, P, P_new)
        hist_new = jnp.concatenate([hist[1:], xv])
        return (hist_new, w_upd, P_upd, n + 1), pred


class OnlineVAR(Transformer):
    """Online vector autoregressive model, TIRSO of Zaman et al. 2019.

    Every sample is a dict of ``N`` channels; the model predicts the
    whole vector at time ``t`` from the previous ``order`` vectors. The
    underlying update is a recursive least squares on the stacked
    history with an exponential forgetting factor ``lam`` and an
    optional group-lasso penalty ``group_lambda`` on each channel-lag
    block. With ``group_lambda = 0`` the model is plain RLS.

    Args:
        order: number of past vectors used in the prediction.
        lam: forgetting factor in ``(0, 1]``.
        group_lambda: group-lasso strength. ``0.0`` disables it.
        alpha: base step size of the RLS update, used as
            ``alpha / (1 + n)`` at step ``n`` (the ``O(1/t)`` schedule
            of Theorem 4 in the source paper).
        delta: initial value of the inverse-Hessian matrix.
        feature_keys: explicit channel order. ``None`` sorts the keys
            of the first dict seen.

    Examples:
        >>> from dense_armor.utility.timeseries import OnlineVAR
        >>> var = OnlineVAR(order=1, lam=1.0, feature_keys=["a", "b"])
        >>> for i in range(30):
        ...     _ = var.learn_one({"a": float(i), "b": float(2 * i)})
        >>> out = var.predict_one({"a": 29.0, "b": 58.0})
        >>> sorted(out.keys())
        ['a', 'b']
    """

    budget_s: float | None = 1e-3
    memory_class: str = "O(1)"

    def __init__(
        self,
        order: int = 1,
        lam: float = 1.0,
        group_lambda: float = 0.0,
        alpha: float = 0.1,
        delta: float = 1e6,
        feature_keys: Sequence[str] | None = None,
    ) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if not 0.0 < lam <= 1.0:
            raise ValueError(f"lam must be in (0, 1], got {lam}")
        if group_lambda < 0.0:
            raise ValueError(f"group_lambda must be >= 0, got {group_lambda}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        self.order = int(order)
        self.lam = float(lam)
        self.group_lambda = float(group_lambda)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.feature_keys = feature_keys
        self._keys: list | None = None
        self._hist: np.ndarray | None = None
        self._phi: np.ndarray | None = None
        self._r: np.ndarray | None = None
        self._a: np.ndarray | None = None
        self._t: int = 0
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        """Count of samples skipped because of a missing or bad value."""
        return self.n_missing_

    def _resolve_keys(self, x: dict) -> list:
        if self._keys is not None:
            return self._keys
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        return self._keys

    def _vec(self, x: dict) -> np.ndarray | None:
        if not isinstance(x, dict):
            return None
        keys = self._resolve_keys(x)
        try:
            v = np.array([float(x[k]) for k in keys], dtype=float)
        except (TypeError, ValueError, KeyError):
            return None
        if not np.all(np.isfinite(v)):
            return None
        return v

    def _init(self, n: int) -> None:
        np_dim = n * self.order
        self._hist = np.zeros((self.order, n))
        self._phi = self.delta * np.eye(np_dim)
        self._r = np.zeros((n, np_dim))
        self._a = np.zeros((n, np_dim))

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "OnlineVAR":
        """One sample: predict, update, push to history."""
        v = self._vec(x)
        if v is None:
            self.n_missing_ += 1
            return self
        if self._phi is None:
            self._init(v.size)
        assert self._hist is not None
        assert self._phi is not None
        assert self._r is not None
        assert self._a is not None
        g = self._hist.flatten()
        self._phi = self.lam * self._phi + np.outer(g, g)
        self._r = self.lam * self._r + np.outer(v, g)
        grad = self._a @ self._phi.T - self._r
        step = self.alpha / (1.0 + self._t)
        a_f = self._a - step * grad
        if self.group_lambda > 0.0:
            n_ch = v.size
            a_new = a_f.copy()
            for n in range(n_ch):
                for npc in range(n_ch):
                    lo = npc * self.order
                    hi = lo + self.order
                    blk = a_f[n, lo:hi]
                    norm = float(np.linalg.norm(blk))
                    if norm > 0.0:
                        scale = max(
                            0.0,
                            1.0 - step * self.group_lambda / norm,
                        )
                        a_new[n, lo:hi] = blk * scale
            self._a = a_new
        else:
            self._a = a_f
        self._hist = np.concatenate([self._hist[1:], v[None, :]])
        self._t += 1
        return self

    def predict_one(self, x: dict, t: float | None = None) -> dict:
        """Forecast the whole vector one step ahead."""
        keys = self._keys if self._keys is not None else (
            list(self.feature_keys) if self.feature_keys is not None else None
        )
        if keys is None:
            keys = sorted(x) if isinstance(x, dict) else None
        if keys is None:
            return {}
        if self._a is None or self._hist is None:
            return {k: 0.0 for k in keys}
        g = self._hist.flatten()
        y_hat = self._a @ g
        return {k: float(y_hat[i]) for i, k in enumerate(keys)}

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """One-step-ahead residual for every channel."""
        v = self._vec(x)
        if v is None:
            return {}
        keys = self._resolve_keys(x)
        pred = self.predict_one(x, t)
        return {k: float(v[i]) - pred[k] for i, k in enumerate(keys)}
