"""Dimensionality reduction for vision features.

Two transformers, both in the spirit of the library: they read a dict
of floats and return a smaller dict of floats, one sample at a time.

- :class:`RandomProjection` draws a fixed random matrix once, in the
  constructor, and multiplies every incoming vector by it. The Johnson
  and Lindenstrauss (1984) lemma guarantees that the pairwise distances
  are preserved up to a factor ``(1 + eps)`` with high probability,
  provided the output dimension ``k`` is large enough. The version used
  here follows Ghojogh et al. (2021), section 2.2: the elements of the
  matrix are i.i.d. Gaussian with mean zero and variance ``1 / k``, so
  the expected squared norm of a projected vector equals the squared
  norm of the original one. The distortion bound is equation 21 of that
  paper, with ``delta = 2 exp(-(eps^2 - eps^3) k / 4)``.
- :class:`IncrementalPCA` learns the top ``k`` directions of the data
  as they stream in, with Oja's rule generalised to ``k`` components.
  The update follows the one-component form in Balsubramani, Dasgupta,
  Freund (2013), equation (1),

      v_n = v_{n-1} + gamma_n (X_n X_n^T - (v_{n-1}^T X_n X_n^T v_{n-1})
              I_d) v_{n-1},

  For ``k > 1`` the matrix ``V`` (``d x k``) is updated with the same
  step, ``V + gamma_n z (V^T z)^T``, and re-orthonormalised with a QR
  decomposition after each sample: a stochastic subspace (power)
  iteration. The QR makes the correction term of Oja's rule redundant,
  so the multi-component case is this iteration, not a separate rule
  from the paper. The running mean is removed from the input first. The step size is ``gamma_n = c / n`` with ``c``
  chosen by the user or by the default rule. The paper analyses the rate
  of convergence: the potential ``1 - (v . v*)^2 / |v|^2`` goes to zero
  as O(1 / n) when ``c`` is at least ``1 / (2 (lambda_1 - lambda_2))``.
"""

from typing import Any

import numpy as np

from dense_armor.roles import Transformer


class RandomProjection(Transformer):
    """Fixed Gaussian random projection to ``k`` dimensions.

    Args:
        k: output dimension. Must be positive.
        seed: seed of the random generator, so a run is reproducible.
        input_dim: number of input features. If ``None`` (default), the
            matrix is built on the first call, using the keys of the
            first dict in sorted order.

    Raises:
        ValueError: if ``k <= 0``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.reduce import RandomProjection
        >>> rp = RandomProjection(k=3, seed=0)
        >>> x = {f"f{i}": float(i) for i in range(6)}
        >>> out = rp.transform_one(x)
        >>> len(out)
        3
        >>> sorted(out)
        ['p0', 'p1', 'p2']
    """

    budget_s = 0.005
    memory_class = "O(1)"

    def __init__(
        self,
        k: int,
        seed: int | None = None,
        input_dim: int | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k
        self.seed = seed
        self.input_dim = input_dim
        self._rng = np.random.default_rng(seed)
        self._W: np.ndarray | None = None
        self._keys: list[str] | None = None
        self.n_missing_ = 0

    def _reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._W = None
        self._keys = None
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def _ensure(self, x: dict) -> None:
        if self._W is not None:
            return
        keys = sorted(x)
        d = self.input_dim if self.input_dim is not None else len(keys)
        self._keys = keys
        self._W = self._rng.standard_normal((d, self.k)) / np.sqrt(self.k)

    def _to_vec(self, x: dict) -> np.ndarray:
        keys = self._keys if self._keys is not None else sorted(x)
        return np.array([float(x[k]) for k in keys], dtype=float)

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "RandomProjection":
        """No learning: the projection is fixed at construction."""
        self._time_step(t)
        if self._W is None:
            self._ensure(x)
        return self

    def transform_one(
        self, x: dict, t: float | None = None
    ) -> dict[str, float]:
        """Project one feature dict onto the random matrix.

        Args:
            x: dict of float features.
            t: unused, kept for the interface.

        Returns:
            A dict with keys ``p0, ..., p{k-1}``.
        """
        if self._W is None:
            self._ensure(x)
        try:
            v = self._to_vec(x)
        except (KeyError, ValueError, TypeError):
            self.n_missing_ += 1
            return {f"p{j}": 0.0 for j in range(self.k)}
        if not np.isfinite(v).all():
            self.n_missing_ += 1
            return {f"p{j}": 0.0 for j in range(self.k)}
        proj = self._W.T @ v
        return {f"p{j}": float(proj[j]) for j in range(self.k)}


class IncrementalPCA(Transformer):
    """Online PCA: Oja's rule for one component, stochastic subspace iteration with QR for several.

    Args:
        k: number of components.
        lr: if given, the constant ``c`` in ``gamma_n = c / n``. If
            ``None`` (default), uses ``c = 1.0``. See the class docstring
            for the rule the paper suggests.
        eps: numerical guard.

    Raises:
        ValueError: if ``k <= 0`` or ``lr <= 0``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.vision.reduce import IncrementalPCA
        >>> rng = np.random.default_rng(0)
        >>> pca = IncrementalPCA(k=1)
        >>> for _ in range(200):
        ...     _ = pca.learn_one({"a": float(rng.normal(1, 0.1)),
        ...                        "b": float(rng.normal(0, 1))})
        >>> out = pca.transform_one({"a": 1.0, "b": 0.0})
        >>> len(out)
        1
    """

    budget_s = 0.01
    memory_class = "O(1)"

    def __init__(
        self,
        k: int,
        lr: float | None = None,
        eps: float = 1e-9,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if lr is not None and lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        self.k = k
        self.lr = lr
        self.eps = eps
        self._V: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._keys: list[str] | None = None
        self._n = 0
        self.n_missing_ = 0

    def _reset(self) -> None:
        self._V = None
        self._mean = None
        self._keys = None
        self._n = 0
        self.n_missing_ = 0

    def _unit_test_skips(self) -> set:
        return set()

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    @property
    def components_(self) -> np.ndarray | None:
        return self._V

    def _ensure(self, x: dict) -> None:
        if self._V is not None:
            return
        keys = sorted(x)
        self._keys = keys
        d = len(keys)
        self._mean = np.zeros(d)
        rng = np.random.default_rng(0)
        v = rng.standard_normal((d, self.k))
        q, _ = np.linalg.qr(v)
        self._V = q[:, : self.k]

    def _to_vec(self, x: dict) -> np.ndarray:
        keys = self._keys if self._keys is not None else sorted(x)
        return np.array([float(x[k]) for k in keys], dtype=float)

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> "IncrementalPCA":
        """Update the components with one sample.

        Args:
            x: dict of float features.
            y: unused.
            t: unused.

        Returns:
            ``self``.
        """
        self._time_step(t)
        if self._V is None:
            self._ensure(x)
        try:
            v = self._to_vec(x)
        except (KeyError, ValueError, TypeError):
            self.n_missing_ += 1
            return self
        if not np.isfinite(v).all():
            self.n_missing_ += 1
            return self
        self._n += 1
        n = self._n
        mean = self._mean
        V = self._V
        assert mean is not None and V is not None
        mean = mean + (v - mean) / n
        self._mean = mean
        z = v - mean
        c = 1.0 if self.lr is None else float(self.lr)
        gamma = c / n
        a = V.T @ z
        update = np.outer(z, a)
        V_new = V + gamma * update
        q, _ = np.linalg.qr(V_new)
        self._V = q[:, : self.k]
        return self

    def transform_one(
        self, x: dict, t: float | None = None
    ) -> dict[str, float]:
        """Project one feature dict onto the current components.

        Args:
            x: dict of float features.
            t: unused, kept for the interface.

        Returns:
            A dict with keys ``pc0, ..., pc{k-1}``.
        """
        if self._V is None:
            self._ensure(x)
        try:
            v = self._to_vec(x)
        except (KeyError, ValueError, TypeError):
            self.n_missing_ += 1
            return {f"pc{j}": 0.0 for j in range(self.k)}
        if not np.isfinite(v).all():
            self.n_missing_ += 1
            return {f"pc{j}": 0.0 for j in range(self.k)}
        mean = self._mean
        V = self._V
        assert mean is not None and V is not None
        z = v - mean
        proj = V.T @ z
        return {f"pc{j}": float(proj[j]) for j in range(self.k)}
