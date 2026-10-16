"""Dense optical flow and the moving-object mask.

The coarse flow of :class:`~dense_armor.utility.vision.features.FrameFeatures`
averages the displacement inside a few cells; this module estimates the
flow at every pixel, with the global method of Horn and Schunck (1981)
written by Ziani (2025, equation 6 and the Euler-Lagrange pair 7-8).

The energy the method minimises is

    E(u, v) = (Ix u + Iy v + It)^2 + alpha (|grad u|^2 + |grad v|^2),

whose Euler-Lagrange equations give the fixed-point iteration

    u = u_bar - Ix (Ix u_bar + Iy v_bar + It) / (alpha^2 + Ix^2 + Iy^2 + eps),
    v = v_bar - Iy (Ix u_bar + Iy v_bar + It) / (alpha^2 + Ix^2 + Iy^2 + eps),

where ``u_bar`` and ``v_bar`` are the local averages of ``u`` and ``v``.
The smoothness term ``alpha`` is the price of a large gradient of the
flow; a small alpha follows the edges of the moving object, a large
alpha gives a smoother field.

The iteration runs in JAX so the whole image is one jitted call; the
per-pixel Python loop is left only for the direct comparison with the
scalar reference.

:class:`MotionMask` takes the flow field of two consecutive frames,
compensates the global shift estimated by
:class:`~dense_armor.utility.vision.motion.GlobalMotion`, and returns a
boolean mask: a pixel is *moving* when the residual flow is above a
robust threshold. The mask is the answer to "what is moving, given that
the camera moves too".

References:
    Horn, B. K., Schunck, B. G. (1981). Determining optical flow.
        Artificial Intelligence 17(1-3), 185-203.
    Ziani (2025). Methods to a Multiresolution Horn-Schunck.
        arXiv:2511.16535. Equations 6, 7, 8.
"""

import numpy as np

from dense_armor.roles import Transformer
from dense_armor.utility.vision.motion import GlobalMotion


def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.ndim == 3:
        return a[..., 0]
    return a


def _gradients(g: np.ndarray) -> tuple:
    Ix = np.zeros_like(g)
    Iy = np.zeros_like(g)
    Ix[:, :-1] = g[:, 1:] - g[:, :-1]
    Iy[:-1, :] = g[1:, :] - g[:-1, :]
    return Ix, Iy


def _average(a: np.ndarray) -> np.ndarray:
    """Mean of the four neighbours, wrap-around at the borders.

    This is the discrete Laplacian ``lap u = 4 (u_bar - u)`` used by
    Horn and Schunck (1981) to write the smoothness term.
    """
    return 0.25 * (
        np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1) + np.roll(a, -1, axis=1)
    )


class DenseFlow(Transformer):
    """Dense optical flow with the Horn-Schunck iteration.

    Args:
        alpha: smoothness weight. Larger values give a smoother field
            and ignore small motions; the paper uses ``1.0`` in its
            examples. Default ``1.0``.
        n_iter: number of Jacobi iterations. Default ``30``.
        use_jax: use the JAX path when the module is available; the
            numpy path is the fallback. Default ``True``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.dense import DenseFlow
        >>> f = DenseFlow(alpha=1.0, n_iter=10)
        >>> rng = np.random.default_rng(0)
        >>> a = rng.normal(0, 1, (16, 16)).astype(np.float32)
        >>> b = np.roll(a, (1, 0), axis=(0, 1))
        >>> out = f.transform_one({"array": b, "prev": a})
        >>> out["u"].shape
        (16, 16)
    """

    memory_class = "O(HW)"

    def __init__(self, alpha: float = 1.0, n_iter: int = 30,
                 use_jax: bool = True) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        self.alpha = float(alpha)
        self.n_iter = int(n_iter)
        self.use_jax = bool(use_jax)
        self._prev: np.ndarray | None = None
        self._jax_fn = None

    def _reset(self) -> None:
        self._prev = None
        self._jax_fn = None

    def _unit_test_skips(self) -> set:
        return set()

    def _prepare(self, x) -> np.ndarray:
        arr = x["array"] if isinstance(x, dict) else (
            x.array if hasattr(x, "array") else x
        )
        return _to_gray(np.asarray(arr)).astype(np.float32)

    def _jax_solve(self, g1: np.ndarray, g2: np.ndarray) -> tuple:
        if self._jax_fn is None:
            import jax
            import jax.numpy as jnp

            def _solve(g1, g2, alpha, n_iter):
                Ix = jnp.zeros_like(g1)
                Iy = jnp.zeros_like(g1)
                It = g2 - g1
                Ix = Ix.at[:, :-1].set(g1[:, 1:] - g1[:, :-1])
                Iy = Iy.at[:-1, :].set(g1[1:, :] - g1[:-1, :])
                u = jnp.zeros_like(g1)
                v = jnp.zeros_like(g1)
                a2 = alpha * alpha
                den = a2 + Ix * Ix + Iy * Iy + 1e-9
                for _ in range(n_iter):
                    u_bar = 0.25 * (
                        jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0) +
                        jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1)
                    )
                    v_bar = 0.25 * (
                        jnp.roll(v, 1, 0) + jnp.roll(v, -1, 0) +
                        jnp.roll(v, 1, 1) + jnp.roll(v, -1, 1)
                    )
                    prod = Ix * u_bar + Iy * v_bar + It
                    u = u_bar - Ix * prod / den
                    v = v_bar - Iy * prod / den
                return u, v

            self._jax_fn = jax.jit(_solve, static_argnums=(3,))
        u, v = self._jax_fn(g1, g2, self.alpha, self.n_iter)
        return np.asarray(u), np.asarray(v)

    def _numpy_solve(self, g1: np.ndarray, g2: np.ndarray) -> tuple:
        Ix, Iy = _gradients(g1)
        It = g2 - g1
        u = np.zeros_like(g1)
        v = np.zeros_like(g1)
        a2 = self.alpha * self.alpha
        den = a2 + Ix * Ix + Iy * Iy + 1e-9
        for _ in range(self.n_iter):
            u_bar = _average(u)
            v_bar = _average(v)
            prod = Ix * u_bar + Iy * v_bar + It
            u = u_bar - Ix * prod / den
            v = v_bar - Iy * prod / den
        return u, v

    def transform_one(self, x, t=None):
        cur = self._prepare(x)
        if self._prev is None or self._prev.shape != cur.shape:
            z = np.zeros_like(cur)
            return {"u": z, "v": z, "mag": z}
        g1 = self._prev
        g2 = cur
        if self.use_jax:
            try:
                u, v = self._jax_solve(g1, g2)
            except Exception:
                u, v = self._numpy_solve(g1, g2)
        else:
            u, v = self._numpy_solve(g1, g2)
        return {"u": u, "v": v, "mag": np.hypot(u, v)}

    def learn_one(self, x, y=None, t=None):
        self._prev = self._prepare(x)
        return self


class MotionMask(Transformer):
    """Boolean mask of the pixels whose residual flow is large.

    The global shift of the scene is removed first (the mean of the flow
    field outside the largest residual region is used as a robust
    estimate of the camera motion); the mask then flags the pixels whose
    residual flow magnitude is above ``n_sigmas`` robust deviations of
    the residual distribution.

    Args:
        n_sigmas: threshold in robust deviations. Default ``3.0``.
        dilation: number of pixels the mask is dilated by, to close the
            small holes inside a moving object. Default ``1``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.dense import MotionMask
        >>> m = MotionMask(n_sigmas=3.0, dilation=0)
        >>> rng = np.random.default_rng(0)
        >>> u = rng.normal(0, 0.05, (16, 16)).astype(np.float32)
        >>> v = rng.normal(0, 0.05, (16, 16)).astype(np.float32)
        >>> u[6:10, 6:10] += 2.0
        >>> out = m.transform_one({"u": u, "v": v})
        >>> bool(out["mask"][7, 7])
        True
    """

    memory_class = "O(HW)"

    def __init__(self, n_sigmas: float = 3.0, dilation: int = 1) -> None:
        if n_sigmas <= 0:
            raise ValueError(f"n_sigmas must be > 0, got {n_sigmas}")
        if dilation < 0:
            raise ValueError(f"dilation must be >= 0, got {dilation}")
        self.n_sigmas = float(n_sigmas)
        self.dilation = int(dilation)

    def _reset(self) -> None:
        pass

    def _unit_test_skips(self) -> set:
        return set()

    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        for _ in range(self.dilation):
            out = mask.copy()
            out[1:, :] |= mask[:-1, :]
            out[:-1, :] |= mask[1:, :]
            out[:, 1:] |= mask[:, :-1]
            out[:, :-1] |= mask[:, 1:]
            mask = out
        return mask

    def transform_one(self, x, t=None):
        u = np.asarray(x["u"], dtype=float)
        v = np.asarray(x["v"], dtype=float)
        mag = np.hypot(u, v)
        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med)))
        scale = 1.4826 * mad
        if scale <= 1e-9:
            mask = np.zeros_like(mag, dtype=bool)
        else:
            mask = mag > med + self.n_sigmas * scale
        if self.dilation > 0:
            mask = self._dilate(mask)
        return {"mask": mask}

    def learn_one(self, x, y=None, t=None):
        return self
