"""Linear Kalman filtering, one sample at a time.

``KalmanFilter`` is the textbook filter with fixed matrices. Two
adaptive variants follow the measurement and process noise.

References:
    Mbalawata, I. S., Sarkka, S., Vihola, M., Haario, H. (2013).
    Adaptive Metropolis algorithm using variational Bayesian adaptive
    Kalman filter. arXiv:1308.5875. Section 3.2 (Variational Noise
    Adaptation): equations (19)-(21) on page 8, Algorithm 1 with
    equation (22) on page 9. The paper used N = 5 fixed-point iterations.

    Popescu, M., Mronga, D., Kumar, S., Kirchner, F. (2026).
    Residual-based adaptive Kalman filtering for legged robot state
    estimation. arXiv:2608.02316. Equations (10) and (11) define the
    innovation and residual; (13) is the sliding-window covariance;
    (14) and (15) adapt the process noise; (16) adapts the measurement
    noise. The paper finds a window of one second best (section 5.3.2).
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from dense_armor.roles import Transformer


class KalmanFilter(Transformer):
    """Linear Kalman filter with fixed matrices.

    Args:
        F: state transition matrix ``(n, n)``.
        H: measurement matrix ``(m, n)``.
        Q: process noise covariance ``(n, n)``.
        R: measurement noise covariance ``(m, m)``.
        x0: initial state mean, length ``n``. Default zeros.
        P0: initial state covariance ``(n, n)``. Default identity.

    The measurement vector at every call is the sorted values of the
    input dict.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.timeseries import KalmanFilter
        >>> kf = KalmanFilter(F=np.eye(2), H=np.eye(2),
        ...                   Q=1e-3 * np.eye(2), R=1e-2 * np.eye(2))
        >>> for i in range(5):
        ...     _ = kf.learn_one({"a": float(i), "b": float(2 * i)})
        >>> est = kf.predict_one({})
        >>> est.shape
        (2,)
    """

    budget_s: float | None = 1e-3
    memory_class: str = "O(1)"

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> None:
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        n = self.F.shape[0]
        self.x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
        self.P0 = np.eye(n) if P0 is None else np.asarray(P0, dtype=float)
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._innovation: np.ndarray | None = None
        self._S: np.ndarray | None = None
        self.n_missing_: int = 0

    @property
    def n_missing(self) -> int:
        """Count of samples skipped because of a missing or bad value."""
        return self.n_missing_

    def _meas(self, x: dict) -> np.ndarray | None:
        if not isinstance(x, dict):
            return None
        m = self.H.shape[0]
        if not x:
            return np.zeros(m)
        try:
            v = np.array([float(x[k]) for k in sorted(x)], dtype=float)
        except (TypeError, ValueError, KeyError):
            return None
        if v.size != m or not np.all(np.isfinite(v)):
            return None
        return v

    def get_params(self, deep: bool = True) -> dict:
        """Return hyper-parameters as plain lists.

        The arrays are converted to nested lists so that two clones
        compare equal with the plain ``==`` operator.
        """
        out = super().get_params(deep)
        for k in ("F", "H", "Q", "R", "x0", "P0"):
            v = out.get(k)
            if v is not None and hasattr(v, "tolist"):
                out[k] = v.tolist()
        return out

    def _ensure_state(self) -> None:
        if self._x is None:
            self._x = self.x0.copy()
            self._P = self.P0.copy()

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "KalmanFilter":
        """Predict then update with the measurement."""
        self._ensure_state()
        assert self._x is not None
        assert self._P is not None
        x_pred = self.F @ self._x
        P_pred = self.F @ self._P @ self.F.T + self.Q
        z = self._meas(x)
        if z is None:
            self.n_missing_ += 1
            self._x = x_pred
            self._P = P_pred
            return self
        y_inn = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self._x = x_pred + K @ y_inn
        self._P = (np.eye(self._P.shape[0]) - K @ self.H) @ P_pred
        self._P = 0.5 * (self._P + self._P.T)
        self._innovation = y_inn
        self._S = S
        return self

    def predict_one(self, x: dict | None = None, t: float | None = None) -> np.ndarray:
        """Return the current state estimate."""
        self._ensure_state()
        assert self._x is not None
        return self._x.copy()

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        """Innovation and its covariance, as dict entries."""
        if self._innovation is None or self._S is None:
            m = self.H.shape[0]
            return {"innovation": np.zeros(m), "S": np.zeros((m, m))}
        return {"innovation": self._innovation.copy(), "S": self._S.copy()}

    def _unit_test_skips(self) -> set:
        """Skip the pure check: state holds vectors, not scalars."""
        return {"check_step_is_pure"}

    def init_state(self, dim: int = 1) -> tuple:
        """Return the initial ``(x, P)`` state as JAX arrays.

        The argument ``dim`` is accepted for symmetry with the other
        time-series estimators and is ignored: the state size is
        ``F.shape[0]``, fixed at construction.
        """
        return (jnp.asarray(self.x0), jnp.asarray(self.P0))

    def step(self, state: tuple, v: Any) -> tuple:
        """One pure Kalman step: ``(state, z) -> (state, x_new[0])``.

        The measurement ``z`` is either a vector of length ``m = H.shape[0]``
        or a scalar, which is broadcast to that length. The returned
        output is the first component of the updated state, so the
        function can be used inside ``jax.lax.scan`` and checked by
        ``check_step_is_pure``.
        """
        x, P = state
        m = self.H.shape[0]
        z = jnp.asarray(v)
        if z.ndim == 0:
            z = jnp.broadcast_to(z, (m,))
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        y_inn = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ jnp.linalg.inv(S)
        x_new = x_pred + K @ y_inn
        P_new = (jnp.eye(P.shape[0]) - K @ self.H) @ P_pred
        P_new = 0.5 * (P_new + P_new.T)
        return (x_new, P_new), x_new[0]


class VBAdaptiveKalman(KalmanFilter):
    """Variational Bayesian adaptive Kalman filter.

    Puts an inverse-Wishart distribution on the measurement noise
    covariance (equation 19 of Mbalawata et al., 2013). Its parameters
    ``nu`` (degrees of freedom) and ``V`` (scale matrix) are updated at
    every measurement by ``n_iter`` fixed-point iterations of
    Algorithm 1 (page 9), started from the previous covariance
    ``V / (nu - d - 1)`` as the algorithm says (``Sigma^(1) = Sigma^-``,
    with ``B = I``).

    The covariance is written through ``V``: after each iteration
    ``V = rho * V_prev + e e' + H P H'`` and ``R = V / (nu - d - 1)``,
    with ``nu = rho * (nu_prev - d - 1) + d + 2``. For ``rho = 1`` this
    is equation (22). For ``rho < 1`` equation (22) as printed weights the
    previous covariance by ``nu_prev - d - 1`` and not by
    ``rho * (nu_prev - d - 1)``; this class uses the second form, which is
    the mean of the inverse-Wishart distribution of equation (20) after the
    forgetting of equation (21). On a stream whose measurement variance
    jumps from 0.01 to 1.0 after 200 samples, 400 samples in all, the
    printed equation gives 2.177 at ``rho = 0.99`` and this class 0.811
    (``rho = 1``: 0.510 for both); that comparison is this class's own
    computation, not a statement of the paper.

    Args:
        rho: forgetting factor on the degrees of freedom, in ``(0, 1]``.
            ``1.0`` is the value the convergence proof of the paper
            assumes (with ``B = I``); the default ``0.99`` forgets old
            measurements.
        n_iter: number of fixed-point iterations per measurement.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        rho: float = 0.99,
        n_iter: int = 5,
    ) -> None:
        super().__init__(F, H, Q, R, x0, P0)
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        self.rho = float(rho)
        self.n_iter = int(n_iter)
        m = self.R.shape[0]
        self._nu: float = m + 2.0
        self._V: np.ndarray = (self._nu - m - 1.0) * self.R.copy()

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "VBAdaptiveKalman":
        """Predict, then update state and R with the VB-AKF iteration."""
        self._ensure_state()
        assert self._x is not None
        assert self._P is not None
        x_pred = self.F @ self._x
        P_pred = self.F @ self._P @ self.F.T + self.Q
        z = self._meas(x)
        if z is None:
            self.n_missing_ += 1
            self._x = x_pred
            self._P = P_pred
            return self
        m = self.R.shape[0]
        H = self.H
        V_prev = self._V
        nu = self.rho * (self._nu - m - 1.0) + m + 2.0
        R_cur = V_prev / (self._nu - m - 1.0)
        x_cur = x_pred.copy()
        P_cur = P_pred.copy()
        V = V_prev
        for _ in range(self.n_iter):
            S = H @ P_pred @ H.T + R_cur
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_cur = x_pred + K @ (z - H @ x_pred)
            P_cur = P_pred - K @ S @ K.T
            P_cur = 0.5 * (P_cur + P_cur.T)
            inn = z - H @ x_cur
            V = self.rho * V_prev + np.outer(inn, inn) + H @ P_cur @ H.T
            R_cur = V / (nu - m - 1.0)
        self._nu = nu
        self._V = V
        self._x = x_cur
        self._P = P_cur
        self._innovation = z - H @ x_cur
        self._S = H @ P_cur @ H.T + R_cur
        return self

    @property
    def R_estimate(self) -> np.ndarray:
        """Current estimate of the measurement noise covariance."""
        m = self.R.shape[0]
        return self._V / (self._nu - m - 1.0)

    def _unit_test_skips(self) -> set:
        """Skip the pure check: the state holds vectors, not scalars."""
        return {"check_step_is_pure"}

    def init_state(self, dim: int = 1) -> tuple:
        """Return ``(x, P, nu, V)`` as JAX arrays.

        ``nu`` is a scalar, ``V`` a ``(m, m)`` matrix. The argument
        ``dim`` is ignored.
        """
        return (
            jnp.asarray(self.x0),
            jnp.asarray(self.P0),
            jnp.asarray(self._nu),
            jnp.asarray(self._V),
        )

    def step(self, state: tuple, v: Any) -> tuple:
        """One pure VB-AKF step with ``n_iter`` fixed-point iterations."""
        x, P, nu_prev, V_prev = state
        m = self.H.shape[0]
        z = jnp.asarray(v)
        if z.ndim == 0:
            z = jnp.broadcast_to(z, (m,))
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        nu = self.rho * (nu_prev - m - 1.0) + m + 2.0
        R_cur = V_prev / (nu_prev - m - 1.0)
        x_cur = x_pred
        P_cur = P_pred
        V = V_prev
        for _ in range(self.n_iter):
            S = self.H @ P_pred @ self.H.T + R_cur
            K = P_pred @ self.H.T @ jnp.linalg.inv(S)
            x_cur = x_pred + K @ (z - self.H @ x_pred)
            P_cur = P_pred - K @ S @ K.T
            P_cur = 0.5 * (P_cur + P_cur.T)
            inn = z - self.H @ x_cur
            V = self.rho * V_prev + jnp.outer(inn, inn) + self.H @ P_cur @ self.H.T
            R_cur = V / (nu - m - 1.0)
        return (x_cur, P_cur, nu, V), x_cur[0]


class InnovationAdaptiveKalman(KalmanFilter):
    """Adaptive Kalman filter after Popescu et al. (2026).

    Adapts ``Q`` and ``R`` from the innovation and residual of the
    filter over a sliding window of ``window`` samples: the residual
    covariance gives ``R`` (equation 16) and the innovation covariance
    gives a scale ``alpha`` for ``Q`` (equations 14 and 15). The adapted
    matrices live in private attributes; ``Q`` and ``R`` keep the values
    given to the constructor, so ``clone()`` and ``get_params()`` return
    the starting filter, and ``R_estimate`` / ``Q_estimate`` return the
    adapted ones.

    Differences from the paper, kept on purpose:

    - ``R`` is forced diagonal (the paper keeps the full matrix of
      equation 16) with a floor of ``1e-9`` on the diagonal.
    - ``alpha`` is clipped to ``[q_scale_min, q_scale_max]``; the paper
      has no clip.
    - ``Q_t = Q * sqrt(alpha)`` scales the whole ``Q``; the paper scales
      only the IMU noise terms of its legged-robot filter.
    - The rotation ``R_WB`` of equation 16 and the average of ``alpha_i``
      over the legs in contact (equation 14) do not apply to a plain
      linear filter and are left out.

    ``step`` is the same filter in pure form: ``Q`` and ``R`` are part
    of the state, so a whole stream through ``jax.lax.scan`` gives the
    same state as ``learn_one`` up to float32 round-off.

    Args:
        window: number of samples in the covariance window. The paper
            recommends one second of samples; the default of 50 matches
            50 Hz.
        q_scale_min: lower clip of the process-noise scaling factor.
        q_scale_max: upper clip of the process-noise scaling factor.
    """

    memory_class: str = "O(window)"

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        window: int = 50,
        q_scale_min: float = 0.1,
        q_scale_max: float = 10.0,
    ) -> None:
        super().__init__(F, H, Q, R, x0, P0)
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = int(window)
        self.q_scale_min = float(q_scale_min)
        self.q_scale_max = float(q_scale_max)
        m = self.R.shape[0]
        self._Q = self.Q.copy()
        self._R = self.R.copy()
        self._buf_inn = np.zeros((self.window, m))
        self._buf_res = np.zeros((self.window, m))
        self._fill = 0
        self._head = 0

    @property
    def R_estimate(self) -> np.ndarray:
        """Current adapted measurement noise covariance."""
        return self._R.copy()

    @property
    def Q_estimate(self) -> np.ndarray:
        """Current adapted process noise covariance."""
        return self._Q.copy()

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "InnovationAdaptiveKalman":
        """One sample with Q and R adapted from the innovation and residual."""
        self._ensure_state()
        assert self._x is not None
        assert self._P is not None
        x_pred = self.F @ self._x
        P_pred = self.F @ self._P @ self.F.T + self._Q
        z = self._meas(x)
        if z is None:
            self.n_missing_ += 1
            self._x = x_pred
            self._P = P_pred
            return self
        H = self.H
        inn = z - H @ x_pred
        S = H @ P_pred @ H.T + self._R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ inn
        P_upd = (np.eye(P_pred.shape[0]) - K @ H) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)
        res = z - H @ x_upd
        self._buf_inn[self._head] = inn
        self._buf_res[self._head] = res
        self._head = (self._head + 1) % self.window
        self._fill = min(self._fill + 1, self.window)
        n = self._fill
        if n >= 2:
            br = self._buf_res[:n]
            R_new = br.T @ br / n + H @ P_upd @ H.T
            R_new = 0.5 * (R_new + R_new.T)
            self._R = np.diag(np.maximum(np.diag(R_new), 1e-9))
            bi = self._buf_inn[:n]
            num = float(np.trace(bi.T @ bi / n - self._R))
            den = float(np.trace(H @ P_pred @ H.T))
            if den > 0.0:
                alpha = min(max(num / den, self.q_scale_min), self.q_scale_max)
                self._Q = self.Q * float(np.sqrt(max(alpha, 0.0)))
                self._Q = 0.5 * (self._Q + self._Q.T)
        self._x = x_upd
        self._P = P_upd
        self._innovation = inn
        self._S = S
        return self

    def _unit_test_skips(self) -> set:
        """Skip the pure check: the state holds vectors, not scalars."""
        return {"check_step_is_pure"}

    def init_state(self, dim: int = 1) -> tuple:
        """Return ``(x, P, buf_inn, buf_res, fill, head, R, Q)``.

        The two ring buffers store the last ``window`` innovation and
        residual vectors, ``fill`` how many entries are populated,
        ``head`` the next write index; ``R`` and ``Q`` are the adapted
        covariances, starting from the constructor values.
        """
        m = self.R.shape[0]
        return (
            jnp.asarray(self.x0),
            jnp.asarray(self.P0),
            jnp.zeros((self.window, m)),
            jnp.zeros((self.window, m)),
            jnp.int32(0),
            jnp.int32(0),
            jnp.asarray(self.R),
            jnp.asarray(self.Q),
        )

    def step(self, state: tuple, v: Any) -> tuple:
        """One pure innovation-adaptive step, equal to ``learn_one``."""
        x, P, buf_inn, buf_res, fill, head, R, Q = state
        m = self.R.shape[0]
        z = jnp.asarray(v)
        if z.ndim == 0:
            z = jnp.broadcast_to(z, (m,))
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + Q
        inn = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + R
        K = P_pred @ self.H.T @ jnp.linalg.inv(S)
        x_upd = x_pred + K @ inn
        P_upd = (jnp.eye(P_pred.shape[0]) - K @ self.H) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)
        res = z - self.H @ x_upd
        buf_inn = buf_inn.at[head].set(inn)
        buf_res = buf_res.at[head].set(res)
        head_new = (head + 1) % self.window
        fill_new = jnp.minimum(fill + 1, self.window)
        mask = (jnp.arange(self.window) < fill_new).astype(buf_res.dtype)[:, None]
        n = fill_new.astype(buf_res.dtype)
        R_raw = (buf_res * mask).T @ buf_res / n + self.H @ P_upd @ self.H.T
        R_raw = 0.5 * (R_raw + R_raw.T)
        R_new = jnp.diag(jnp.maximum(jnp.diag(R_raw), 1e-9))
        adapt = fill_new >= 2
        R_out = jnp.where(adapt, R_new, R)
        C_inn = (buf_inn * mask).T @ buf_inn / n
        num = jnp.trace(C_inn - R_out)
        den = jnp.trace(self.H @ P_pred @ self.H.T)
        alpha = jnp.clip(
            num / jnp.maximum(den, 1e-30), self.q_scale_min, self.q_scale_max
        )
        Q_new = jnp.asarray(self.Q) * jnp.sqrt(jnp.maximum(alpha, 0.0))
        Q_new = 0.5 * (Q_new + Q_new.T)
        Q_out = jnp.where(adapt & (den > 0.0), Q_new, Q)
        return (
            x_upd,
            P_upd,
            buf_inn,
            buf_res,
            fill_new,
            head_new,
            R_out,
            Q_out,
        ), x_upd[0]
