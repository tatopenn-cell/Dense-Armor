"""Field-aware factorization machines, from Juan et al. (2016)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Classifier, Regressor


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + float(np.exp(-z)))
    e = float(np.exp(z))
    return e / (1.0 + e)


class _FFMBase:
    """Shared code for FFMClassifier and FFMRegressor.

    The model is eq. (4) of Juan et al. (2016), page 3::

        y_hat(x) = sum_{j1 < j2} <w_{j1,f2}, w_{j2,f1}> x_{j1} x_{j2}

    where `f1` and `f2` are the fields of `j1` and `j2`. Unlike the
    plain factorization machine there is no `O(k n)` shortcut: each
    pair uses a different slice of the `w` tensor, so the direct
    computation is `O(n_bar^2 k)` (page 3, right after eq. (4)) and
    that is the one implemented here.

    The paper does not include the bias and the linear terms (page 2,
    the paragraph before section 3): the model is exactly the pairwise
    term above, my choice to stay with the paper.

    The factors are initialised from `U[0, 1 / sqrt(k)]` (page 3, below
    Algorithm 1), not from the `N(0, sigma^2)` used for the plain FM.

    The update is AdaGrad per coordinate (Algorithm 1, page 3). The
    accumulator `G` starts at one (page 3, "The initial values of G are
    set to one in order to prevent a large value"), so no `1 / sqrt(0)`.

    Args:
        fields: mapping from feature name to field name. Every feature
            that appears in ``learn_one`` must be in this map; a
            feature that is not, is ignored (no factor update).
        k: number of factors, strictly positive.
        lr: AdaGrad learning rate (``eta`` in the paper), strictly
            positive.
        l2: L2 regularization strength (``lambda`` in the paper),
            non-negative.
        feature_keys: explicit list of feature names. ``None`` sorts
            the keys of the first dict seen.
        seed: seed of the initialiser.

    Raises:
        ValueError: if ``k < 1``, ``lr <= 0`` or ``l2 < 0``.
    """

    budget_s: float = 1e-2
    memory_class: str = "O(n * f * k)"

    def __init__(
        self,
        fields: dict[str, str],
        k: int = 4,
        lr: float = 0.1,
        l2: float = 0.0,
        feature_keys: Sequence[str] | None = None,
        seed: int = 0,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if l2 < 0.0:
            raise ValueError(f"l2 must be >= 0, got {l2}")
        self.fields = dict(fields)
        self.k = int(k)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.feature_keys = feature_keys
        self.seed = int(seed)
        self._keys: list[str] | None = None
        self._key_index: dict[str, int] | None = None
        self._field_names: list[str] | None = None
        self._field_index: dict[str, int] | None = None
        self._w: np.ndarray | None = None
        self._G: np.ndarray | None = None
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
        fields_seen = [self.fields[k] for k in self._keys if k in self.fields]
        self._field_names = sorted(set(fields_seen))
        self._field_index = {f: i for i, f in enumerate(self._field_names)}
        return self._keys

    def _ensure_state(self) -> None:
        if self._w is not None:
            return
        assert self._keys is not None
        assert self._field_names is not None
        n = len(self._keys)
        f = len(self._field_names)
        rng = np.random.default_rng(self.seed)
        self._w = rng.uniform(0.0, 1.0 / np.sqrt(self.k), size=(n, f, self.k))
        self._G = np.ones((n, f, self.k), dtype=float)

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
        if self._w is None:
            return 0.0
        active = self._active(x)
        score = 0.0
        for a, (i, fi, xi) in enumerate(active):
            for j, fj, xj in active[a + 1 :]:
                score += float(np.dot(self._w[i, fj], self._w[j, fi])) * xi * xj
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

    def _grad_error(self, score: float, y: Any) -> float:
        raise NotImplementedError

    def _learn_common(self, x: Any, y: Any) -> None:
        if self._nan_in(x, y):
            self.n_missing += 1
            return
        keys = self._resolve_keys(x)
        assert keys is not None
        self._ensure_state()
        assert self._w is not None
        assert self._G is not None
        score = self._y_hat(x)
        err = self._grad_error(score, y)
        active = self._active(x)
        for a, (i, fi, xi) in enumerate(active):
            for j, fj, xj in active[a + 1 :]:
                if self.l2 > 0.0:
                    g_i = self.l2 * self._w[i, fj] + err * self._w[j, fi] * xi * xj
                    g_j = self.l2 * self._w[j, fi] + err * self._w[i, fj] * xi * xj
                else:
                    g_i = err * self._w[j, fi] * xi * xj
                    g_j = err * self._w[i, fj] * xi * xj
                self._G[i, fj] += g_i * g_i
                self._G[j, fi] += g_j * g_j
                self._w[i, fj] -= self.lr * g_i / np.sqrt(self._G[i, fj])
                self._w[j, fi] -= self.lr * g_j / np.sqrt(self._G[j, fi])
        self.n_seen_ += 1


class FFMClassifier(_FFMBase, Classifier):
    """Field-aware factorization machine for binary classification.

    The model is eq. (4) of Juan et al. (2016), page 3::

        y_hat(x) = sum_{j1 < j2} <w_{j1, f2}, w_{j2, f1}> x_{j1} x_{j2}

    with `f1` the field of `j1` and `f2` the field of `j2`. Each feature
    has `f` latent vectors of length `k`, one for every field, so the
    parameters are `n * f * k`. There is no `O(k n)` shortcut: the
    prediction is `O(n_bar^2 k)` (page 3, Table 2).

    The training is the logit loss of eq. (1) (page 1) solved with
    AdaGrad (Algorithm 1, page 3): the per-coordinate accumulator `G`
    starts at one and grows with the squared sub-gradients of eq. (5)
    and eq. (6). The factor initialiser is `U[0, 1 / sqrt(k)]`
    (page 3, below Algorithm 1).

    Args:
        fields: mapping from feature name to field name.
        k: number of factors per field.
        lr: AdaGrad learning rate.
        l2: L2 strength.
        feature_keys: explicit list of feature names.
        seed: seed of the initialiser.

    Examples:
        >>> from dense_armor.utility.facto.ffm import FFMClassifier
        >>> fields = {"a": "F1", "b": "F1", "c": "F2"}
        >>> ffm = FFMClassifier(fields=fields, k=2, lr=0.1, seed=0)
        >>> for _ in range(500):
        ...     _ = ffm.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 1)
        >>> round(ffm.predict_proba_one({"a": 1.0, "b": 0.5, "c": 0.3})[1], 3) > 0.5
        True

    References:
        Juan, Y., Zhuang, Y., Chin, W.-S., Lin, C.-J. (2016). Field-aware
        Factorization Machines for CTR Prediction. In RecSys 2016.
        Eq. (4) page 3 (model), eq. (5)-(6) page 3 (gradients),
        Algorithm 1 page 3 (AdaGrad), initialiser and `G` start
        page 3 (below Algorithm 1).
    """

    def _grad_error(self, score: float, y: Any) -> float:
        yv = float(y)
        return -yv / (1.0 + float(np.exp(yv * score)))

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "FFMClassifier":
        """One AdaGrad step on the logit loss of a single sample."""
        self._learn_common(x, y)
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """Predicted probabilities of the two classes.

        Before any ``learn_one`` call returns ``{}``.
        """
        if self._w is None:
            return {}
        p = _sigmoid(self._y_hat(x))
        return {0: 1.0 - p, 1: p}

    def predict_one(self, x: dict, t: float | None = None):
        """Predicted class, or ``None`` before any ``learn_one`` call."""
        if self._w is None:
            return None
        return 1 if _sigmoid(self._y_hat(x)) >= 0.5 else 0


class FFMRegressor(_FFMBase, Regressor):
    """Field-aware factorization machine with squared loss.

    The model is eq. (4) of Juan et al. (2016), page 3, the same as
    :class:`FFMClassifier`. The paper is written for binary
    classification (CTR); reusing the model with squared loss and the
    update of Algorithm 1 is my choice, because the task asks for a
    learner of the same family and the gradient on the score with
    squared loss is `score - y`.

    Args:
        fields: mapping from feature name to field name.
        k: number of factors per field.
        lr: AdaGrad learning rate.
        l2: L2 strength.
        feature_keys: explicit list of feature names.
        seed: seed of the initialiser.

    Examples:
        >>> from dense_armor.utility.facto.ffm import FFMRegressor
        >>> fields = {"a": "F1", "b": "F1", "c": "F2"}
        >>> ffm = FFMRegressor(fields=fields, k=2, lr=0.1, seed=0)
        >>> for _ in range(500):
        ...     _ = ffm.learn_one({"a": 1.0, "b": 0.5, "c": 0.3}, 1.0)
        >>> ffm.predict_one({"a": 1.0, "b": 0.5, "c": 0.3}) > 0.5
        True

    References:
        Juan, Y., Zhuang, Y., Chin, W.-S., Lin, C.-J. (2016). Field-aware
        Factorization Machines for CTR Prediction. In RecSys 2016.
        Eq. (4) page 3; Algorithm 1 page 3 for the update.
    """

    def _grad_error(self, score: float, y: Any) -> float:
        return score - float(y)

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> "FFMRegressor":
        """One AdaGrad step on the squared loss of a single sample."""
        self._learn_common(x, y)
        return self

    def predict_one(
        self, x: dict, t: float | None = None, return_std: bool = False
    ) -> float:
        """Predict the target of a single sample.

        Before any ``learn_one`` call returns ``0.0``.
        """
        return self._y_hat(x)
