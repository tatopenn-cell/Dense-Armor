"""Field-weighted factorization machines, from Pan et al. (2018)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Classifier, Regressor
from dense_armor.utility.facto._common import _sigmoid


class _FwFMBase:
    """Shared code for FwFMClassifier and FwFMRegressor.

    The model is eq. (7) of Pan et al. (2018), page 3::

        y_hat(x) = w0
                 + sum_i w_i x_i
                 + sum_{i<j} <v_i, v_j> x_i x_j r_{F(i), F(j)}

    with `F(i)` the field of feature `i` and `r_{F(i), F(j)}` a scalar
    weight that captures the interaction strength of the field pair
    `(F(i), F(j))`. The weights `r` are symmetric (the paper says the
    pair `(F(i), F(j))` is unordered), so only the upper triangle is
    read; the lower triangle is ignored.

    Initialiser (my choice): `v_i ~ N(0, 1e-2)` and `r = 1`. The paper
    says (Section 4, page 3) only that `r` is a learned scalar per
    field pair; it does not prescribe the initialiser, and starting
    `r` at one makes the model equal to a plain FM at initialisation,
    so the learning rate applies to a known baseline.

    L2 (my choice): the same strength applies to `w0`, `w`, `V` and
    `r`. The paper puts L2 on the whole objective (eq. (1), page 2);
    whether `w0` is included is not stated.

    Args:
        optimizer: any optimizer from ``dense_armor.utility.optim``.
            The state is updated once per sample on the concatenated
            parameter vector ``[w0, w, V.ravel(), R.ravel()]``.
        fields: mapping from feature name to field name.
        k: number of latent factors, strictly positive.
        l2: L2 strength, non-negative.
        feature_keys: explicit list of feature names. ``None`` sorts
            the keys of the first dict seen.
        seed: seed of the initialiser.

    Raises:
        ValueError: if ``k < 1`` or ``l2 < 0``.
    """

    budget_s: float = 2e-3
    memory_class: str = "O(n * k + f * f)"

    def __init__(
        self,
        optimizer: Any,
        fields: dict[str, str],
        k: int = 4,
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
        self.fields = dict(fields)
        self.k = int(k)
        self.l2 = float(l2)
        self.feature_keys = feature_keys
        self.seed = int(seed)
        self.init_scale = float(init_scale)
        self._keys: list[str] | None = None
        self._key_index: dict[str, int] | None = None
        self._field_names: list[str] | None = None
        self._field_index: dict[str, int] | None = None
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
        seen = [self.fields[k] for k in self._keys if k in self.fields]
        self._field_names = sorted(set(seen))
        self._field_index = {f: i for i, f in enumerate(self._field_names)}
        return self._keys

    def _ensure_state(self) -> None:
        if self._theta is not None:
            return
        assert self._keys is not None
        assert self._field_names is not None
        n = len(self._keys)
        f = len(self._field_names)
        rng = np.random.default_rng(self.seed)
        w0 = np.zeros(1)
        w = np.zeros(n)
        V = rng.normal(0.0, self.init_scale, size=(n, self.k))
        R = np.ones((f, f))
        self._theta = np.concatenate([w0, w, V.ravel(), R.ravel()])

    def _unpack(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        assert self._theta is not None
        assert self._keys is not None
        assert self._field_names is not None
        n = len(self._keys)
        f = len(self._field_names)
        w0 = float(self._theta[0])
        w = self._theta[1 : 1 + n]
        base = 1 + n
        V = self._theta[base : base + n * self.k].reshape(n, self.k)
        R = self._theta[base + n * self.k :].reshape(f, f)
        return w0, w, V, R

    def _active(self, x: Any) -> list[tuple[int, int, float]]:
        xv = self._as_dict(x)
        assert self._key_index is not None
        assert self._field_index is not None
        out = []
        for key, val in xv.items():
            idx = self._key_index.get(key)
            if idx is None:
                continue
            field = self.fields.get(key)
            if field is None:
                continue
            f_idx = self._field_index.get(field)
            if f_idx is None:
                continue
            fv = float(val)
            if fv != 0.0:
                out.append((idx, f_idx, fv))
        return out

    def _y_hat(self, x: Any) -> float:
        if self._theta is None:
            return 0.0
        w0, w, V, R = self._unpack()
        active = self._active(x)
        score = w0
        for a, (i, fi, xi) in enumerate(active):
            score += w[i] * xi
            for j, fj, xj in active[a + 1 :]:
                lo, hi = (fi, fj) if fi <= fj else (fj, fi)
                score += float(np.dot(V[i], V[j])) * xi * xj * R[lo, hi]
        return float(score)

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
        _w0, _w, V, R = self._unpack()
        score = self._y_hat(x)
        err = self._loss_grad(score, y)
        active = self._active(x)
        grad = np.zeros_like(self._theta)
        assert self._keys is not None
        assert self._field_names is not None
        n = len(self._keys)
        f = len(self._field_names)
        base_v = 1 + n
        base_r = base_v + n * self.k
        grad[0] = err
        for a, (i, fi, xi) in enumerate(active):
            grad[1 + i] += err * xi
            for j, fj, xj in active[a + 1 :]:
                lo, hi = (fi, fj) if fi <= fj else (fj, fi)
                dot = float(np.dot(V[i], V[j]))
                pair = err * xi * xj
                r_ij = R[lo, hi]
                grad[base_v + i * self.k : base_v + (i + 1) * self.k] += (
                    pair * r_ij * V[j]
                )
                grad[base_v + j * self.k : base_v + (j + 1) * self.k] += (
                    pair * r_ij * V[i]
                )
                grad[base_r + lo * f + hi] += err * dot * xi * xj
        if self.l2 > 0.0:
            grad = grad + self.l2 * self._theta
        self._theta = self.optimizer.step(self._theta, grad)
        self.n_seen_ += 1


class FwFMClassifier(_FwFMBase, Classifier):
    """Field-weighted factorization machine for binary classification.

    The model is eq. (7) of Pan et al. (2018), page 3::

        y_hat = w0 + sum_i w_i x_i
              + sum_{i<j} <v_i, v_j> x_i x_j r_{F(i), F(j)}

    The predicted probability of class 1 is ``sigmoid(y_hat)``. The loss
    is binary log-loss, so the gradient on the score is
    ``sigmoid(y_hat) - y``.

    Args:
        optimizer: any optimizer of ``dense_armor.utility.optim``.
        fields: mapping from feature name to field name.
        k: number of latent factors.
        l2: L2 strength.
        feature_keys: explicit list of feature names.
        seed: seed of the initialiser.

    Examples:
        >>> from dense_armor.utility.optim import SGD
        >>> from dense_armor.utility.facto.fwfm import FwFMClassifier
        >>> fields = {"a": "F1", "b": "F1", "c": "F2"}
        >>> m = FwFMClassifier(optimizer=SGD(lr=0.1), fields=fields, k=2, seed=0)
        >>> for _ in range(500):
        ...     _ = m.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 1)
        >>> round(m.predict_proba_one({"a": 1.0, "b": 0.5, "c": 0.3})[1], 3) > 0.5
        True

    References:
        Pan, J., Xu, J., Ruiz, A. L., Zhao, W., Pan, S., Sun, Y., Lu, Q.
        (2018). Field-weighted Factorization Machines for Click-Through
        Rate Prediction in Display Advertising. In WWW 2018.
        Eq. (7) page 3 (model), Section 4 page 3 (the `r` weight).
    """

    def _loss_grad(self, score: float, y: Any) -> float:
        return _sigmoid(score) - float(y)

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "FwFMClassifier":
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


class FwFMRegressor(_FwFMBase, Regressor):
    """Field-weighted factorization machine with squared loss.

    The model is eq. (7) of Pan et al. (2018), page 3, same as
    :class:`FwFMClassifier`. The paper is written for binary
    classification (CTR); reusing the model with squared loss and the
    gradient ``score - y`` is my choice, requested by the task.

    Args:
        optimizer: any optimizer of ``dense_armor.utility.optim``.
        fields: mapping from feature name to field name.
        k: number of latent factors.
        l2: L2 strength.
        feature_keys: explicit list of feature names.
        seed: seed of the initialiser.

    Examples:
        >>> from dense_armor.utility.optim import SGD
        >>> from dense_armor.utility.facto.fwfm import FwFMRegressor
        >>> fields = {"a": "F1", "b": "F1", "c": "F2"}
        >>> m = FwFMRegressor(optimizer=SGD(lr=0.1), fields=fields, k=2, seed=0)
        >>> for _ in range(500):
        ...     _ = m.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 1.0)
        >>> m.predict_one({"a": 1.0, "b": 0.5, "c": 0.3}) > 0.5
        True

    References:
        Pan, J. et al. (2018). Field-weighted Factorization Machines for
        Click-Through Rate Prediction in Display Advertising. In WWW 2018.
        Eq. (7) page 3.
    """

    def _loss_grad(self, score: float, y: Any) -> float:
        return score - float(y)

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "FwFMRegressor":
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
