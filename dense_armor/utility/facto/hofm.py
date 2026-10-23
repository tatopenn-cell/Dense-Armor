"""Higher-order factorization machines, from Blondel et al. (2016)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Classifier, Regressor
from dense_armor.utility.facto._common import _sigmoid


def _anova_forward(
    p: np.ndarray, idx: np.ndarray, vals: np.ndarray, m: int
) -> np.ndarray:
    """Algorithm 1, page 4: A_t for t = 1..m on the support of x.

    Returns an array of shape ``(nz + 1, m + 1)``, ``a[j, t]`` is
    ``A_t(p_{1:j}, x_{1:j})`` for ``j`` from 0 to ``nz`` (number of
    non-zero features) and ``t`` from 0 to ``m``. ``a[j, 0] = 1`` and
    ``a[j, t] = 0`` for ``j < t``.
    """
    nz = len(idx)
    a = np.zeros((nz + 1, m + 1))
    a[:, 0] = 1.0
    for t in range(1, m + 1):
        for j in range(t, nz + 1):
            a[j, t] = a[j - 1, t] + p[idx[j - 1]] * vals[j - 1] * a[j - 1, t - 1]
    return a


def _anova_grad(p: np.ndarray, idx: np.ndarray, vals: np.ndarray, m: int) -> np.ndarray:
    """Algorithm 2, page 4: gradient of A_m w.r.t. p, on the support.

    Returns an array of the same shape as ``p``, zero outside the
    support of ``x`` (page 4: the gradient and x have the same support).
    """
    nz = len(idx)
    a = _anova_forward(p, idx, vals, m)
    atilde = np.zeros((nz + 1, m + 2))
    atilde[nz, m] = 1.0
    for t in range(m, 0, -1):
        for j in range(nz - 1, t - 1, -1):
            atilde[j, t] = atilde[j + 1, t] + atilde[j + 1, t + 1] * p[idx[j]] * vals[j]
    out = np.zeros_like(p)
    for j in range(1, nz + 1):
        s = 0.0
        for t in range(1, m + 1):
            s += atilde[j, t] * a[j - 1, t - 1] * vals[j - 1]
        out[idx[j - 1]] = s
    return out


class _HOFMBase:
    """Shared code for HOFMClassifier and HOFMRegressor.

    The model is eq. (3) and eq. (6) of Blondel et al. (2016), page 2::

        y_hat(x) = <w, x>
                 + sum_{t=2}^{m} sum_{s=1}^{k_t} A_t(p_s^{(t)}, x)

    where `A_t` is the ANOVA kernel of degree `t` (eq. (5), page 2),
    computed with the dynamic programme of Algorithm 1 (page 4) in
    `O(nz(x) t)` time. The gradient of `A_t` with respect to `p` is
    computed with Algorithm 2 (page 4) by reverse-mode differentiation,
    in `O(nz(x) t)` time.

    The parameters `k_2, ..., k_m` are taken equal to a single `k`
    (page 2: "To avoid the combinatorial explosion of hyper-parameter
    combinations to search, in our experiments we will simply set
    beta_1 = ... = beta_m and k_2 = ... = k_m").

    The order `t = 1` is not a degree of the model: eq. (6) starts the
    sum at `t = 2` and puts the linear part in `<w, x>`.

    Initialiser (my choice): `w = 0`, `P^{(t)} ~ N(0, scale^2)`. The
    paper does not prescribe the initialiser.

    Args:
        optimizer: any optimizer of ``dense_armor.utility.optim``.
        m: maximum degree of the ANOVA kernel, at least 2.
        k: number of latent factors, shared across degrees.
        l2: L2 strength, non-negative.
        feature_keys: explicit list of feature names. ``None`` sorts
            the keys of the first dict seen.
        seed: seed of the initialiser.
        init_scale: standard deviation of the initialiser.

    Raises:
        ValueError: if ``m < 2``, ``k < 1`` or ``l2 < 0``.
    """

    budget_s: float = 1.5e-2
    memory_class: str = "O(d + (m - 1) * d * k)"

    def __init__(
        self,
        optimizer: Any,
        m: int = 3,
        k: int = 4,
        l2: float = 0.0,
        feature_keys: Sequence[str] | None = None,
        seed: int = 0,
        init_scale: float = 0.01,
    ) -> None:
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        self.optimizer = optimizer
        self.m = int(m)
        self.k = int(k)
        self.l2 = float(l2)
        self.feature_keys = feature_keys
        self.seed = int(seed)
        self.init_scale = float(init_scale)
        self._keys: list[str] | None = None
        self._key_index: dict[str, int] | None = None
        self._theta: np.ndarray | None = None
        self.n_missing_: int = 0
        self.n_seen_: int = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

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

    def _ensure_state(self) -> None:
        if self._theta is not None:
            return
        assert self._keys is not None
        d = len(self._keys)
        rng = np.random.default_rng(self.seed)
        w = np.zeros(d)
        P_blocks = []
        for _ in range(self.m - 1):
            P_blocks.append(rng.normal(0.0, self.init_scale, size=(d, self.k)))
        self._theta = np.concatenate([w] + [P.ravel() for P in P_blocks])

    def _w(self) -> np.ndarray:
        assert self._theta is not None
        assert self._keys is not None
        d = len(self._keys)
        return self._theta[:d]

    def _block(self, t: int) -> np.ndarray:
        """Block of P^{(t)} for degree t = 2..m, of shape (d, k)."""
        assert self._theta is not None
        assert self._keys is not None
        d = len(self._keys)
        start = d + (t - 2) * d * self.k
        return self._theta[start : start + d * self.k].reshape(d, self.k)

    def _active(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        xv = self._as_dict(x)
        assert self._key_index is not None
        idx_list = []
        val_list = []
        for key, val in xv.items():
            i = self._key_index.get(key)
            if i is None:
                continue
            fv = float(val)
            if fv != 0.0:
                idx_list.append(i)
                val_list.append(fv)
        return np.asarray(idx_list, dtype=int), np.asarray(val_list, dtype=float)

    def _y_hat(self, x: Any) -> float:
        if self._theta is None:
            return 0.0
        idx, vals = self._active(x)
        w = self._w()
        score = float(w[idx] @ vals)
        for t in range(2, self.m + 1):
            P = self._block(t)
            for s in range(self.k):
                p_s = P[:, s]
                a = _anova_forward(p_s, idx, vals, t)
                score += float(a[len(idx), t])
        return score

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

    def _learn_common(self, x: Any, y: Any) -> None:
        if self._nan_in(x, y):
            self.n_missing_ += 1
            return
        self._resolve_keys(x)
        self._ensure_state()
        assert self._theta is not None
        assert self._keys is not None
        idx, vals = self._active(x)
        score = self._y_hat(x)
        err = self._loss_grad(score, y)
        d = len(self._keys)
        grad = np.zeros_like(self._theta)
        for i, v in zip(idx, vals):
            grad[i] += err * v
        for t in range(2, self.m + 1):
            P = self._block(t)
            start = d + (t - 2) * d * self.k
            for s in range(self.k):
                p_s = P[:, s]
                g_p = _anova_grad(p_s, idx, vals, t)
                grad[start + s : start + s + d * self.k : self.k] += err * g_p
        if self.l2 > 0.0:
            grad = grad + self.l2 * self._theta
        self._theta = self.optimizer.step(self._theta, grad)
        self.n_seen_ += 1


class HOFMRegressor(_HOFMBase, Regressor):
    """Higher-order factorization machine for regression.

    The model is eq. (3) and eq. (6) of Blondel et al. (2016), page 2,
    the same as :class:`HOFMClassifier` with squared loss.

    Args:
        optimizer: any optimizer of ``dense_armor.utility.optim``.
        m: maximum degree, at least 2.
        k: number of latent factors, shared across degrees.
        l2: L2 strength.
        feature_keys: explicit list of feature names.
        seed: seed of the initialiser.

    Examples:
        >>> from dense_armor.utility.optim import SGD
        >>> from dense_armor.utility.facto.hofm import HOFMRegressor
        >>> reg = HOFMRegressor(optimizer=SGD(lr=0.05), m=3, k=2, seed=0)
        >>> for _ in range(300):
        ...     _ = reg.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 1.5)
        >>> abs(reg.predict_one({"a": 1.0, "b": 0.5, "c": 0.3}) - 1.5) < 1.0
        True

    References:
        Blondel, M., Fujino, A., Ueda, N., Ishihata, M. (2016).
        Higher-Order Factorization Machines. In NIPS 2016.
        Eq. (5)-(6) page 2 (model), Algorithm 1 and 2 page 4 (DP).
    """

    def _loss_grad(self, score: float, y: Any) -> float:
        return score - float(y)

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "HOFMRegressor":
        """One gradient step on the squared loss of a single sample."""
        self._learn_common(x, y)
        return self

    def predict_one(
        self, x: dict, t: float | None = None, return_std: bool = False
    ) -> float:
        """Predict the target of a single sample.

        Before any ``learn_one`` call returns ``0.0``.
        """
        return self._y_hat(x)


class HOFMClassifier(_HOFMBase, Classifier):
    """Higher-order factorization machine for binary classification.

    The model is eq. (3) and eq. (6) of Blondel et al. (2016), page 2.
    The predicted probability of class 1 is ``sigmoid(y_hat)`` and the
    loss is binary log-loss, so the gradient on the score is
    ``sigmoid(y_hat) - y``.

    Args:
        optimizer: any optimizer of ``dense_armor.utility.optim``.
        m: maximum degree, at least 2.
        k: number of latent factors, shared across degrees.
        l2: L2 strength.
        feature_keys: explicit list of feature names.
        seed: seed of the initialiser.

    Examples:
        >>> from dense_armor.utility.optim import SGD
        >>> from dense_armor.utility.facto.hofm import HOFMClassifier
        >>> clf = HOFMClassifier(optimizer=SGD(lr=0.05), m=3, k=2, seed=0)
        >>> for _ in range(300):
        ...     _ = clf.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 1)
        >>> round(clf.predict_proba_one({"a": 1.0, "b": 0.5, "c": 0.3})[1], 2) > 0.5
        True

    References:
        Blondel, M., Fujino, A., Ueda, N., Ishihata, M. (2016).
        Higher-Order Factorization Machines. In NIPS 2016.
        Eq. (5)-(6) page 2, Algorithm 1 and 2 page 4.
    """

    def _loss_grad(self, score: float, y: Any) -> float:
        return _sigmoid(score) - float(y)

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "HOFMClassifier":
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
