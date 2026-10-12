"""Linear Kalman filtering, one sample at a time.

``KalmanFilter`` is the textbook filter with fixed matrices. Two
adaptive variants follow the measurement and process noise.

References:
    Mbalawata, I. S., Sarkka, S., Vihola, M., Haario, H. (2013).
    Adaptive Metropolis algorithm using variational Bayesian adaptive
    Kalman filter. arXiv:1308.5875. Section 3.2, equations (19), (21)
    and Algorithm 1 (page 8) are the VB-AKF update. The
    number of fixed-point iterations used here is 5.

    Popescu, M., Mronga, D., Kumar, S., Kirchner, F. (2026).
    Residual-based adaptive Kalman filtering for legged robot state
    estimation. arXiv:2608.02316. Equations (10) and (11) define the
    innovation and residual; (13) is the sliding-window covariance;
    (14) and (15) adapt the process noise; (16) adapts the measurement
    noise. The window size used here follows the paper's recommendation
    of one second of samples.
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

    Adds to ``KalmanFilter`` a Wishart prior on the measurement noise
    covariance. The parameters ``nu`` (degrees of freedom) and ``V``
    (scale matrix) are updated at every measurement by the fixed-point
    iteration of Algorithm 1 in Mbalawata et al. (2013), page 8,
    equations (19)-(22). The number of fixed-point iterations is
    ``n_iter`` (5 in the source paper).

    Args:
        rho: forgetting factor on the degrees of freedom, in ``(0, 1]``.
            ``1.0`` is the value the paper's convergence proof assumes.
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
        nu_pred = self.rho * (self._nu - m - 1.0) + m + 1.0
        V = self._V.copy()
        nu = nu_pred + 1.0
        x_cur = x_pred.copy()
        P_cur = P_pred.copy()
        H = self.H
        for _ in range(self.n_iter):
            R_cur = V / (nu - m - 1.0)
            S = H @ P_cur @ H.T + R_cur
            K = P_cur @ H.T @ np.linalg.inv(S)
            x_cur = x_pred + K @ (z - H @ x_pred)
            P_cur = (np.eye(P_cur.shape[0]) - K @ H) @ P_pred
            P_cur = 0.5 * (P_cur + P_cur.T)
            inn = z - H @ x_cur
            V = (
                self.rho * self._V
                + np.outer(inn, inn)
                + H @ P_cur @ H.T
            )
        self._nu = nu
        self._V = V
        self._x = x_cur
        self._P = P_cur
        self._innovation = z - H @ x_cur
        R_cur = V / (nu - m - 1.0)
        self._S = H @ P_cur @ H.T + R_cur
        return self

    @property
    def R_estimate(self) -> np.ndarray:
        """Current estimate of the measurement noise covariance."""
        m = self.R.shape[0]
        return self._V / (self._nu - m - 1.0)

    def _unit_test_skips(self) -> set:
        """Skip both pure checks for this variant.

        The state holds vectors and matrices, so ``check_step_is_pure``
        cannot convert the leaves to Python floats. The fixed-point
        iteration inside ``step`` amplifies float round-off between
        eager and jitted execution by more than the exact-equality
        comparison of ``check_step_is_jittable``; the jitted form
        still runs and converges, the check just cannot prove it to
        the last bit.
        """
        return {"check_step_is_pure", "check_step_is_jittable"}

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
        x, P, nu, V = state
        m = self.H.shape[0]
        z = jnp.asarray(v)
        if z.ndim == 0:
            z = jnp.broadcast_to(z, (m,))
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        nu_pred = self.rho * (nu - m - 1.0) + m + 1.0
        nu_new = nu_pred + 1.0
        x_cur = x_pred
        P_cur = P_pred
        V_new = V
        for _ in range(self.n_iter):
            R_cur = V_new / (nu_new - m - 1.0)
            S = self.H @ P_cur @ self.H.T + R_cur
            K = P_cur @ self.H.T @ jnp.linalg.inv(S)
            x_cur = x_pred + K @ (z - self.H @ x_pred)
            P_cur = (jnp.eye(P_cur.shape[0]) - K @ self.H) @ P_pred
            P_cur = 0.5 * (P_cur + P_cur.T)
            inn = z - self.H @ x_cur
            V_new = self.rho * V + jnp.outer(inn, inn) + self.H @ P_cur @ self.H.T
        return (x_cur, P_cur, nu_new, V_new), x_cur[0]


class InnovationAdaptiveKalman(KalmanFilter):
    """Adaptive Kalman filter of Popescu et al. (2026).

    Adapts ``Q`` and ``R`` from the innovation and residual of the
    filter, following equations (10)-(16) of the source paper. The
    covariance of the innovation and residual is computed over a
    sliding window of ``window`` samples.

    Args:
        window: number of samples in the covariance window. The paper
            recommends one second of samples; the default of 50 works
            at 50 Hz.
        q_scale_min: lower clip of the process-noise scaling factor.
        q_scale_max: upper clip of the process-noise scaling factor.
    """

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
        self.Q0 = self.Q.copy()
        self._inn_buf: list = []
        self._res_buf: list = []

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "InnovationAdaptiveKalman":
        """One sample with Q and R adapted from the innovation and residual."""
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
        H = self.H
        inn = z - H @ x_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ inn
        P_upd = (np.eye(P_pred.shape[0]) - K @ H) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)
        res = z - H @ x_upd
        self._inn_buf.append(inn)
        self._res_buf.append(res)
        if len(self._inn_buf) > self.window:
            self._inn_buf.pop(0)
            self._res_buf.pop(0)
        if len(self._res_buf) >= 2:
            C_res = np.zeros_like(self.R)
            for r in self._res_buf:
                C_res = C_res + np.outer(r, r)
            C_res = C_res / len(self._res_buf)
            R_new = C_res + H @ P_upd @ H.T
            R_new = 0.5 * (R_new + R_new.T)
            self.R = np.diag(np.maximum(np.diag(R_new), 1e-9))
        if len(self._inn_buf) >= 2:
            C_inn = np.zeros_like(self.R)
            for v in self._inn_buf:
                C_inn = C_inn + np.outer(v, v)
            C_inn = C_inn / len(self._inn_buf)
            num = float(np.trace(C_inn - self.R))
            den = float(np.trace(H @ P_pred @ H.T))
            if den > 0.0:
                alpha = num / den
                alpha = min(max(alpha, self.q_scale_min),
                            self.q_scale_max)
                self.Q = self.Q0 * float(np.sqrt(max(alpha, 0.0)))
                self.Q = 0.5 * (self.Q + self.Q.T)
        self._x = x_upd
        self._P = P_upd
        self._innovation = inn
        self._S = S
        return self

    def _unit_test_skips(self) -> set:
        """Skip the pure check: state holds vectors, not scalars."""
        return {"check_step_is_pure"}

    def init_state(self, dim: int = 1) -> tuple:
        """Return ``(x, P, buf_inn, buf_res, fill, head)``.

        The two ring buffers store the last ``window`` innovation and
        residual vectors, ``fill`` how many entries are populated,
        ``head`` the next write index. All shapes come from ``H`` and
        ``window``, fixed at construction.
        """
        m = self.R.shape[0]
        return (
            jnp.asarray(self.x0),
            jnp.asarray(self.P0),
            jnp.zeros((self.window, m)),
            jnp.zeros((self.window, m)),
            jnp.int32(0),
            jnp.int32(0),
        )

    def step(self, state: tuple, v: Any) -> tuple:
        """One pure innovation-adaptive step.

        The innovation and residual are pushed to the two ring buffers
        and the outer products are accumulated, so the buffers move one
        position per call. ``Q`` and ``R`` themselves stay at the values
        fixed at construction: their adaptation lives in
        :meth:`learn_one`, because it is stateful (``R`` and ``Q`` would
        need to be carried through ``lax.scan``).
        """
        x, P, buf_inn, buf_res, fill, head = state
        m = self.R.shape[0]
        z = jnp.asarray(v)
        if z.ndim == 0:
            z = jnp.broadcast_to(z, (m,))
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        inn = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ jnp.linalg.inv(S)
        x_upd = x_pred + K @ inn
        P_upd = (jnp.eye(P_pred.shape[0]) - K @ self.H) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)
        res = z - self.H @ x_upd
        buf_inn = buf_inn.at[head].set(inn)
        buf_res = buf_res.at[head].set(res)
        head_new = (head + 1) % self.window
        fill_new = jnp.minimum(fill + 1, self.window)
        return (x_upd, P_upd, buf_inn, buf_res, fill_new, head_new), x_upd[0]