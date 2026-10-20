"""Factorization machines of degree 2, from Rendle (2010)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Classifier, Regressor
from dense_armor.utility.facto._common import _sigmoid


class _FMBase:
    """Common code for FMRegressor and FMClassifier."""

    budget_s = 1.5e-2
    memory_class: str = "O(n * k)"

    def __init__(
        self,
        optimizer: Any,
        k: int = 8,
        l2: float = 0.0,
        feature_keys: Sequence[str] | None = None,
        seed: int = 0,
        init_scale: float = 0.01,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        self.optimizer = optimizer
        self.k = int(k)
        self.l2 = float(l2)
        self.feature_keys = feature_keys
        self.seed = int(seed)
        self.init_scale = float(init_scale)
        self._keys: list[str] | None = None
        self._key_index: dict[str, int] | None = None
        self._theta: np.ndarray | None = None
        self.n_missing: int = 0
        self.n_seen_: int = 0

    @staticmethod
    def _as_dict(x: Any) -> dict:
        if isinstance(x, dict):
            return x
        to_dict = getattr(x, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        raise TypeError(
            f"expected a dict or an object with to_dict(), got {type(x).__name__}"
        )

    def _resolve_keys(self, x: Any) -> list[str]:
        x = self._as_dict(x)
        if self._keys is not None:
            return self._keys
        if self.feature_keys is not None:
            self._keys = list(self.feature_keys)
        else:
            self._keys = sorted(x)
        self._key_index = {key: i for i, key in enumerate(self._keys)}
        return self._keys

    def _ensure_state(self, keys: list) -> None:
        if self._theta is not None:
            return
        n = len(keys)
        rng = np.random.default_rng(self.seed)
        theta = np.zeros(1 + n + n * self.k, dtype=float)
        theta[1 + n :] = rng.normal(0.0, self.init_scale, size=n * self.k)
        self._theta = theta

    def _unpack(self) -> tuple[float, np.ndarray, np.ndarray]:
        assert self._theta is not None
        n = len(self._keys) if self._keys is not None else 0
        w0 = float(self._theta[0])
        w = self._theta[1 : 1 + n]
        V = self._theta[1 + n :].reshape(n, self.k)
        return w0, w, V

    def _active(self, x: Any) -> list[tuple[int, float]]:
        xv = self._as_dict(x)
        assert self._key_index is not None
        out = []
        for key, val in xv.items():
            idx = self._key_index.get(key)
            if idx is None:
                continue
            fv = float(val)
            if fv != 0.0:
                out.append((idx, fv))
        return out

    def _y_hat(self, x: Any) -> float:
        if self._theta is None:
            return 0.0
        w0, w, V = self._unpack()
        active = self._active(x)
        linear = w0
        s = np.zeros(self.k)
        sq = np.zeros(self.k)
        for i, val in active:
            linear += w[i] * val
            s += V[i] * val
            sq += (V[i] ** 2) * (val * val)
        return float(linear + 0.5 * np.sum(s * s - sq))

    def _pairwise_naive(self, x: Any) -> float:
        if self._theta is None:
            return 0.0
        w0, w, V = self._unpack()
        active = self._active(x)
        linear = w0
        pair = 0.0
        for i, (ii, vi) in enumerate(active):
            linear += w[ii] * vi
            for jj, vj in active[i + 1 :]:
                pair += float(np.dot(V[ii], V[jj])) * vi * vj
        return float(linear + pair)

    def _nan_in(self, x: Any, y: Any) -> bool:
        xv = self._as_dict(x)
        if not xv:
            return True
        for val in xv.values():
            try:
                fv = float(val)
            except (TypeError, ValueError):
                return True
            if not np.isfinite(fv):
                return True
        try:
            yv = float(y)
        except (TypeError, ValueError):
            return True
        return not np.isfinite(yv)

    def _loss_grad(self, score: float, y: Any) -> float:
        raise NotImplementedError

    def _compute_gradient(self, x: Any, err: float) -> np.ndarray:
        assert self._theta is not None
        n = len(self._keys) if self._keys is not None else 0
        _, _, V = self._unpack()
        active = self._active(x)
        s = np.zeros(self.k)
        for i, val in active:
            s += V[i] * val
        grad = np.zeros_like(self._theta)
        grad[0] = err
        for i, val in active:
            grad[1 + i] = err * val
            start = 1 + n + i * self.k
            grad[start : start + self.k] = err * (val * (s - V[i] * val))
        if self.l2 > 0.0:
            grad = grad + self.l2 * self._theta
        return grad

    def _learn_common(self, x: Any, y: Any) -> None:
        if self._nan_in(x, y):
            self.n_missing += 1
            return
        keys = self._resolve_keys(x)
        self._ensure_state(keys)
        score = self._y_hat(x)
        err = self._loss_grad(score, y)
        grad = self._compute_gradient(x, err)
        self._theta = self.optimizer.step(self._theta, grad)
        self.n_seen_ += 1


class FMRegressor(_FMBase, Regressor):
    """Factorization machine of degree 2, for regression.

    The model equation is eq. (1) of Rendle (2010), page 2::

        y_hat(x) = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j

    with `<v_i, v_j> := sum_f v_{i,f} v_{j,f}` (eq. (3), page 2) and the
    parameters `w0 in R`, `w in R^n`, `V in R^{n x k}` (eq. (2), page 2).
    `k` is the dimension of the factorization, `n` the number of features.

    The pairwise term is computed in `O(k n)` and not in `O(k n^2)`, with
    the rewriting of Lemma 3.1, page 3::

        sum_{i<j} <v_i, v_j> x_i x_j
            = 1/2 sum_f ( (sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2 )

    The gradient of the model is eq. (4), page 3::

        d/dw0    y_hat = 1
        d/dw_i   y_hat = x_i
        d/dv_{i,f} y_hat = x_i (sum_j v_{j,f} x_j) - v_{i,f} x_i^2

    The factor vectors are initialised from `N(0, init_scale^2)`. Rendle
    does not prescribe the initialisation (Section III-C, page 3, only
    says that V is learned by gradient descent), so the choice is mine.

    The L2 penalty also acts on `w0`. Rendle does not exclude the global
    bias from the L2 term but does not prescribe the opposite either, so
    the choice is mine.

    Args:
        optimizer: any optimizer of ``dense_armor.utility.optim``. The
            state is updated once per sample on the concatenated
            parameter vector ``[w0, w, V.flatten()]``, so optimizers
            with per-parameter state (Adam, AdaGrad, FTRL) work.
        k: number of factors, strictly positive.
        l2: L2 regularization strength, ``0.0`` for none.
        feature_keys: explicit list of feature keys. ``None`` sorts the
            keys of the first dict seen.
        seed: seed of the initialiser.
        init_scale: standard deviation of the initialiser.

    Raises:
        ValueError: if ``k < 1`` or ``l2 < 0``.

    Examples:
        >>> from dense_armor.utility.optim import SGD
        >>> from dense_armor.utility.facto.fm import FMRegressor
        >>> fm = FMRegressor(optimizer=SGD(lr=0.05), k=2, seed=0)
        >>> for _ in range(500):
        ...     _ = fm.learn_one({"a": 1.0, "b": 0.5}, 2.0)
        >>> abs(fm.predict_one({"a": 1.0, "b": 0.5}) - 2.0) < 0.5
        True

    References:
        Rendle, S. (2010). Factorization Machines. In ICDM 2010.
        Eq. (1), (2), (3) page 2; Lemma 3.1 and eq. (4) page 3.
    """

    def _loss_grad(self, score: float, y: Any) -> float:
        return score - float(y)

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "FMRegressor":
        """One gradient step on the squared loss of a single sample."""
        self._learn_common(x, y)
        return self

    def predict_one(
        self, x: dict, t: float | None = None, return_std: bool = False
    ) -> float:
        """Predict the target of a single sample.

        Before any ``learn_one`` call the prediction is ``0.0``.
        """
        return self._y_hat(x)


class FMClassifier(_FMBase, Classifier):
    """Factorization machine of degree 2, for binary classification.

    The score is eq. (1) of Rendle (2010), page 2 (same as
    :class:`FMRegressor`), and the predicted probability of class 1 is
    ``sigmoid(score)``. The loss is binary log-loss, so the gradient on
    the score is ``sigmoid(score) - y``. Rendle (Section III-B, page 3)
    states that binary classification uses the sign of ``y_hat`` and
    the logit loss; the sigmoid parametrisation is the same model with
    the probability explicit, my choice.

    Args:
        optimizer: any optimizer of ``dense_armor.utility.optim``.
        k: number of factors, strictly positive.
        l2: L2 regularization strength, ``0.0`` for none.
        feature_keys: explicit list of feature keys. ``None`` sorts the
            keys of the first dict seen.
        seed: seed of the initialiser.
        init_scale: standard deviation of the initialiser.

    Raises:
        ValueError: if ``k < 1`` or ``l2 < 0``.

    Examples:
        >>> from dense_armor.utility.optim import SGD
        >>> from dense_armor.utility.facto.fm import FMClassifier
        >>> fc = FMClassifier(optimizer=SGD(lr=0.05), k=2, seed=0)
        >>> for _ in range(500):
        ...     _ = fc.learn_one({"a": 1.0, "b": 0.5}, 1)
        >>> p = fc.predict_proba_one({"a": 1.0, "b": 0.5})
        >>> p[1] > 0.5
        True

    References:
        Rendle, S. (2010). Factorization Machines. In ICDM 2010.
        Eq. (1), (2), (3) page 2; Lemma 3.1 and eq. (4) page 3.
    """

    def _loss_grad(self, score: float, y: Any) -> float:
        return _sigmoid(score) - float(y)

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "FMClassifier":
        """One gradient step on the log-loss of a single sample."""
        self._learn_common(x, y)
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Predicted probabilities of the two classes.

        Before any ``learn_one`` call returns ``{}``.
        """
        if self._theta is None:
            return {}
        p = _sigmoid(self._y_hat(x))
        return {0: 1.0 - p, 1: p}

    def predict_one(self, x: dict, t: float | None = None):
        """Predicted class, or ``None`` before any ``learn_one`` call."""
        if self._theta is None:
            return None
        return 1 if _sigmoid(self._y_hat(x)) >= 0.5 else 0
