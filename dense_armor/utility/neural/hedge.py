"""Hedge Backpropagation (Sahoo et al. 2018).

``HedgeNetClassifier`` implements Algorithm 1 (page 4) of Sahoo, Pham,
Lu and Hoi, *Online Deep Learning: Learning Deep Neural Networks on
the Fly*, arXiv:1711.03705: ``L + 1`` classifiers on the ``L + 1``
representations ``h(0) = x, h(l) = relu(W(l) h(l-1))``, each with a
softmax output ``f(l) = softmax(h(l) Theta(l))``; the prediction is
``F(x) = sum_l alpha(l) f(l)(x)`` (Eq.(1), page 3), the classifier
weights are updated by Eq.(2) and the representation weights by Eq.(3)
(page 4), the ``alphas_`` by the Hedge discount ``beta ** L(l)``, then
smoothed to ``s / L`` and normalised (page 4). The hidden activation
is relu (the paper writes a generic ``sigma``), the Hedge loss is
``1 - p(l)[y]`` (the paper asks for a value in ``(0, 1)`` and does not
give it), the gradients use the cross-entropy of each classifier: all
our choices, stated in the docstring.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.roles import Classifier
from dense_armor.utility.neural.online import _EqWrap, _unwrap
from dense_armor.utility.neural.params import Param


def _relu(x: jax.Array) -> jax.Array:
    return jax.nn.relu(x)


class HedgeNetClassifier(Classifier):
    """Online deep learning with Hedge Backpropagation.

    ``sizes=[d, h1, ..., hL]`` gives the input size and the ``L``
    hidden layers. There are ``L + 1`` classifiers: the classifier
    ``l`` reads ``h(l)`` (with ``h(0) = x``) and produces a softmax
    over the classes; the prediction is the ``alphas_``-weighted sum of
    them (Eq.(1), page 3 of Sahoo et al. 2018).

    ``learn_one(x, y)`` computes every classifier's probabilities, then
    updates ``Theta(l)`` by Eq.(2) and ``W(l)`` by Eq.(3) of page 4,
    then the ``alphas_`` by the Hedge discount ``beta ** (1 - p(l)[y])``,
    then the smoothing ``max(alpha, s/L)`` and the normalisation.

    Args:
        sizes: the widths ``[d, h1, ..., hL]``, at least two entries.
        key: a ``jax.random`` key.
        classes: the class labels, at least two.
        beta: the Hedge discount, in ``(0, 1)``; default 0.99.
        lr: the learning rate of the online gradient descent, positive.
        smoothing: the minimum weight parameter ``s``, in ``(0, 1)``;
            default 0.2.
        feature_keys: the keys of the dict read as features, in order.

    Raises:
        ValueError: on ``len(sizes) < 2``, ``len(classes) < 2``, ``lr <= 0``,
            ``beta`` outside ``(0, 1)`` or ``smoothing`` outside ``(0, 1)``.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.hedge import HedgeNetClassifier
        >>> c = HedgeNetClassifier([2, 4], jax.random.PRNGKey(0), classes=[0, 1])
        >>> c.learn_one({"a": 1.0, "b": 0.0}, 1)
        HedgeNetClassifier(n_seen_=1, n_missing_=0, L_=1, classes=[0, 1])
        >>> sorted(c.predict_proba_one({"a": 1.0, "b": 0.0}).keys())
        [0, 1]
    """

    budget_s = 0.01
    memory_class = "O(1)"

    def __init__(
        self,
        sizes: Any,
        key: Any,
        classes: Any,
        beta: float = 0.99,
        lr: float = 0.1,
        smoothing: float = 0.2,
        feature_keys: Any = None,
    ) -> None:
        sizes = list(sizes)
        if len(sizes) < 2:
            raise ValueError(f"sizes needs at least two entries, got {len(sizes)}")
        classes_list = list(classes)
        if len(classes_list) < 2:
            raise ValueError(
                f"classes needs at least two labels, got {len(classes_list)}"
            )
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0 < beta < 1):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if not (0 < smoothing < 1):
            raise ValueError(f"smoothing must be in (0, 1), got {smoothing}")
        self.sizes = sizes
        self.key = _unwrap(key)
        self.classes = classes_list
        self.beta = float(beta)
        self.lr = float(lr)
        self.smoothing = float(smoothing)
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.L_ = len(sizes) - 1
        n_classes = len(classes_list)
        keys = jax.random.split(self.key, self.L_ + 1 + self.L_)
        self.Ws_ = [
            Param(
                jax.random.normal(keys[i], (sizes[i], sizes[i + 1]))
                * math.sqrt(2.0 / sizes[i])
            )
            for i in range(self.L_)
        ]
        self.Thetas_ = [
            Param(
                jax.random.normal(keys[self.L_ + i], (sizes[i], n_classes))
                * math.sqrt(1.0 / sizes[i])
            )
            for i in range(self.L_ + 1)
        ]
        self.alphas_ = jnp.full((self.L_ + 1,), 1.0 / (self.L_ + 1), dtype=jnp.float32)
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.feature_keys_: list[str] | None = None

    def __repr__(self) -> str:
        return (
            f"HedgeNetClassifier(n_seen_={self.n_seen_}, "
            f"n_missing_={self.n_missing_}, L_={self.L_}, "
            f"classes={self.classes!r})"
        )

    def _read(self, x: Any) -> jnp.ndarray | None:
        if not isinstance(x, dict):
            return None
        keys = (
            self.feature_keys if self.feature_keys is not None else self.feature_keys_
        )
        if keys is None:
            return None
        vals: list[float] = []
        for k in keys:
            if k not in x:
                return None
            v = x[k]
            if v is None:
                return None
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(fv):
                return None
            vals.append(fv)
        return jnp.asarray(vals, dtype=jnp.float32)

    def _forward_all(self, vec: jnp.ndarray) -> list[jnp.ndarray]:
        hs = [vec]
        for i in range(self.L_):
            h = _relu(self.Ws_[i].value.T @ hs[-1] + jnp.zeros(self.sizes[i + 1]))
            hs.append(h)
        return hs

    def _probas_from_hs(self, hs: list[jnp.ndarray]) -> list[jnp.ndarray]:
        out = []
        for i in range(self.L_ + 1):
            logits = hs[i] @ self.Thetas_[i].value
            out.append(jax.nn.softmax(logits))
        return out

    def learn_one(self, x: dict, y: Any, t: float | None = None) -> HedgeNetClassifier:
        """One online step of Algorithm 1; returns ``self``."""
        del t
        if y not in self.classes:
            self.n_missing_ += 1
            return self
        if self.feature_keys is None and self.feature_keys_ is None:
            if not isinstance(x, dict):
                self.n_missing_ += 1
                return self
            self.feature_keys_ = sorted(x.keys())
        vec = self._read(x)
        if vec is None:
            self.n_missing_ += 1
            return self
        hs = self._forward_all(vec)
        probs = self._probas_from_hs(hs)
        y_idx = self.classes.index(y)
        n_classes = len(self.classes)
        onehot = jnp.zeros((n_classes,)).at[y_idx].set(1.0)
        dWs = [jnp.zeros_like(w.value) for w in self.Ws_]
        dThetas = [jnp.zeros_like(t.value) for t in self.Thetas_]
        for j in range(self.L_ + 1):
            d_logits = (probs[j] - onehot) * self.alphas_[j]
            dThetas[j] = jnp.outer(hs[j], d_logits)
            d_h = self.Thetas_[j].value @ d_logits
            for l in range(j, 0, -1):
                d_pre = d_h * (hs[l] > 0).astype(d_h.dtype)
                dWs[l - 1] = dWs[l - 1] + jnp.outer(hs[l - 1], d_pre)
                d_h = self.Ws_[l - 1].value @ d_pre
        for i in range(self.L_):
            self.Ws_[i].value = self.Ws_[i].value - self.lr * dWs[i]
        for i in range(self.L_ + 1):
            self.Thetas_[i].value = self.Thetas_[i].value - self.lr * dThetas[i]
        new_alphas = []
        for j in range(self.L_ + 1):
            p_true = float(probs[j][y_idx])
            L_j = 1.0 - p_true
            L_j = min(max(L_j, 1e-12), 1.0 - 1e-12)
            a = float(self.alphas_[j]) * (self.beta**L_j)
            a = max(a, self.smoothing / self.L_)
            new_alphas.append(a)
        alphas_arr = jnp.asarray(new_alphas, dtype=jnp.float32)
        self.alphas_ = alphas_arr / jnp.sum(alphas_arr)
        self.n_seen_ += 1
        return self

    def predict_proba_one(self, x: dict, t: float | None = None) -> dict:
        """The Hedge-weighted class probabilities ``F(x)``."""
        del t
        if self.feature_keys_ is None and self.feature_keys is not None:
            self.feature_keys_ = list(self.feature_keys)
        vec = self._read(x)
        if vec is None:
            return {}
        hs = self._forward_all(vec)
        probs = self._probas_from_hs(hs)
        F = jnp.zeros((len(self.classes),), dtype=jnp.float32)
        for j in range(self.L_ + 1):
            F = F + self.alphas_[j] * probs[j]
        vals = [float(F[i]) for i in range(len(self.classes))]
        s = sum(vals)
        if s > 0.0:
            vals = [v / s for v in vals]
        return {c: vals[i] for i, c in enumerate(self.classes)}

    def predict_one(self, x: dict, t: float | None = None) -> Any:
        """The class with the largest ``F(x)``, or None if the model cannot predict."""
        probs = self.predict_proba_one(x, t=t)
        if not probs:
            return None
        return max(probs, key=lambda k: probs[k])

    def params(self) -> list[Param]:
        """The representation weights followed by the classifier weights."""
        return list(self.Ws_) + list(self.Thetas_)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor parameters, with the key wrapped for equality."""
        d = super().get_params(deep=deep)
        if "key" in d:
            d["key"] = _EqWrap(d["key"])
        return d

    def state_dict(self) -> dict[str, Any]:
        """The weights, the alphas and the counters."""
        return {
            "Ws_": copy.deepcopy(self.Ws_),
            "Thetas_": copy.deepcopy(self.Thetas_),
            "alphas_": np.asarray(self.alphas_).copy(),
            "n_seen_": self.n_seen_,
            "n_missing_": self.n_missing_,
            "feature_keys_": (
                list(self.feature_keys_) if self.feature_keys_ is not None else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> HedgeNetClassifier:
        """Restore from ``state_dict()``."""
        self.Ws_ = copy.deepcopy(state["Ws_"])
        self.Thetas_ = copy.deepcopy(state["Thetas_"])
        self.alphas_ = jnp.asarray(state["alphas_"])
        self.n_seen_ = state["n_seen_"]
        self.n_missing_ = state["n_missing_"]
        self.feature_keys_ = (
            list(state["feature_keys_"]) if state["feature_keys_"] is not None else None
        )
        return self

    @classmethod
    def _unit_test_params(cls):
        import jax

        yield {
            "sizes": (3, 4),
            "key": jax.random.PRNGKey(0),
            "classes": (0, 1),
            "feature_keys": ("a", "b", "c"),
        }

    def _unit_test_skips(self) -> set:
        return set()
