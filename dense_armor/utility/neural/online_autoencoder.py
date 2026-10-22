"""An autoencoder trained online, used as an anomaly detector.

``OnlineAutoencoder`` wraps an ``Autoencoder`` of item A as an
``AnomalyDetector`` of the library: ``score_one(x)`` is the
reconstruction error of the autoencoder on the sample, high =
anomalous, and ``learn_one(x)`` updates the autoencoder one sample at
a time with the same step as ``OnlineNetRegressor`` of the library (the
target is the input itself, ``y = x``). The reconstruction error as
the anomaly score is the argument of Chalapathy and Chawla (2019),
pages 6 and 9: an autoencoder trained on normal samples reconstructs
them with a low error and reconstructs an unusual sample with a high
one.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import jax
import jax.numpy as jnp

from dense_armor.roles import AnomalyDetector
from dense_armor.utility.neural.autoencoder import Autoencoder
from dense_armor.utility.neural.losses import MSE
from dense_armor.utility.neural.online import _EqWrap, _unwrap
from dense_armor.utility.neural.schedules import ConstantLR


def _read(
    x: Any,
    keys: list[str] | None,
) -> tuple[jnp.ndarray | None, list[str] | None]:
    """Read ``x`` into a float32 vector; None on missing or non-finite."""
    if not isinstance(x, dict):
        return None, keys
    use = keys if keys is not None else sorted(x.keys())
    vals: list[float] = []
    for k in use:
        if k not in x:
            return None, use
        v = x[k]
        if v is None:
            return None, use
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None, use
        if not math.isfinite(fv):
            return None, use
        vals.append(fv)
    return jnp.asarray(vals, dtype=jnp.float32), use


class OnlineAutoencoder(AnomalyDetector):
    """An autoencoder trained one sample at a time, as an anomaly score.

    ``learn_one(x, t=None)`` updates the autoencoder with the same
    operations, in the same order, as ``OnlineNetRegressor`` of the library:
    ``zero_grad()`` on every ``Param``, ``forward(vector,
    training=True)``, the ``MSE`` loss against the vector itself, the
    ``backward``, the learning rate ``schedule(n_seen_)`` written to
    every optimizer, the update, ``n_seen_ += 1``. A sample is one
    unit of the schedule (our choice, stated here).

    ``score_one(x, t=None)`` returns the reconstruction error of the
    sample: a ``float``, high = anomalous, and ``0.0`` for an
    incomplete or non-finite sample.

    ``encode_one(x)`` returns the code as a dict
    ``{"c0": ..., "c1": ...}``, one key per output of the encoder.

    Args:
        model: an ``Autoencoder`` of item A (a model with
            ``forward``, ``backward``, ``params``, ``encode``,
            ``decode``, ``reconstruction_error``).
        optimizer: a prototype of ``dense_armor.utility.optim``; one
            ``copy.deepcopy`` for each ``Param``.
        feature_keys: the dict keys read as features, in order; if
            None, the sorted keys of the first dict are used.
        schedule: a learning-rate schedule; default ``ConstantLR``
            with the learning rate of ``optimizer``.

    Raises:
        ValueError: if ``model`` has no ``encode`` or ``decode``
            method.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from dense_armor.utility.neural.autoencoder import Autoencoder
        >>> from dense_armor.utility.neural.chain import Sequential
        >>> from dense_armor.utility.neural.layers import Dense
        >>> from dense_armor.utility.neural.online_autoencoder import OnlineAutoencoder
        >>> from dense_armor.utility.optim import SGD
        >>> enc = Sequential([Dense(2, 1, jax.random.PRNGKey(0))])
        >>> dec = Sequential([Dense(1, 2, jax.random.PRNGKey(1))])
        >>> ae = Autoencoder(enc, dec)
        >>> oa = OnlineAutoencoder(ae, SGD(0.1), feature_keys=("a", "b"))
        >>> isinstance(oa.score_one({"a": 1.0, "b": 0.0}), float)
        True
    """

    budget_s = 0.05
    memory_class = "O(1)"

    def __init__(
        self,
        model: Autoencoder,
        optimizer: Any,
        feature_keys: Any = None,
        schedule: Any = None,
    ) -> None:
        model = _unwrap(model)
        optimizer = _unwrap(optimizer)
        if schedule is not None:
            schedule = _unwrap(schedule)
        if not hasattr(model, "encode") or not hasattr(model, "decode"):
            raise ValueError(
                "model must have encode and decode methods (an Autoencoder)"
            )
        self.model = model
        self.optimizer = optimizer
        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.schedule = schedule if schedule is not None else ConstantLR(optimizer.lr)
        self.model_ = copy.deepcopy(model)
        self.optimizers_ = [copy.deepcopy(optimizer) for _ in model.params()]
        self.loss_ = MSE()
        self.n_seen_ = 0
        self.n_missing_ = 0
        self.feature_keys_: list[str] | None = None

    def __repr__(self) -> str:
        return (
            f"OnlineAutoencoder(n_seen_={self.n_seen_}, n_missing_={self.n_missing_})"
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor parameters, wrapped so two clones compare equal."""
        d = super().get_params(deep=deep)
        for k in ("model", "optimizer", "schedule"):
            if k in d:
                d[k] = _EqWrap(d[k])
        return d

    def learn_one(self, x: dict, t: float | None = None) -> OnlineAutoencoder:
        """One online step; returns ``self``."""
        del t
        if self.feature_keys is None and self.feature_keys_ is None:
            if not isinstance(x, dict):
                self.n_missing_ += 1
                return self
            self.feature_keys_ = sorted(x.keys())
        keys = (
            self.feature_keys if self.feature_keys is not None else self.feature_keys_
        )
        vec, keys_used = _read(x, keys)
        if vec is None:
            self.n_missing_ += 1
            return self
        if self.feature_keys is None:
            self.feature_keys_ = keys_used
        lr_t = self.schedule(float(self.n_seen_))
        for opt in self.optimizers_:
            opt.lr = lr_t
        for p in self.model_.params():
            p.zero_grad()
        pred = self.model_.forward(vec, training=True)
        vec_2d = vec.reshape(-1, 1)
        self.loss_.forward(pred.reshape(-1, 1), vec_2d)
        self.model_.backward(self.loss_.backward().reshape(-1))
        for p, opt in zip(self.model_.params(), self.optimizers_):
            p.update(opt)
        self.n_seen_ += 1
        return self

    def score_one(self, x: dict, t: float | None = None) -> float:
        """The reconstruction error of ``x``; high = anomalous."""
        del t
        keys = (
            self.feature_keys if self.feature_keys is not None else self.feature_keys_
        )
        if keys is None:
            return 0.0
        vec, _ = _read(x, keys)
        if vec is None:
            return 0.0
        err = self.model_.reconstruction_error(vec)
        return float(jnp.asarray(err))

    def encode_one(self, x: dict, t: float | None = None) -> dict:
        """The code of ``x``, one key per output of the encoder."""
        del t
        keys = (
            self.feature_keys if self.feature_keys is not None else self.feature_keys_
        )
        if keys is None:
            return {}
        vec, _ = _read(x, keys)
        if vec is None:
            return {}
        z = self.model_.encode(vec)
        z = jnp.asarray(z).reshape(-1)
        return {f"c{i}": float(z[i]) for i in range(z.shape[0])}

    @classmethod
    def _unit_test_params(cls):

        from dense_armor.utility.neural.autoencoder import Autoencoder
        from dense_armor.utility.neural.chain import Sequential
        from dense_armor.utility.neural.layers import Dense
        from dense_armor.utility.optim import SGD

        enc = Sequential([Dense(3, 2, jax.random.PRNGKey(0))])
        dec = Sequential([Dense(2, 3, jax.random.PRNGKey(1))])
        yield {
            "model": Autoencoder(enc, dec),
            "optimizer": SGD(0.01),
            "feature_keys": ("a", "b", "c"),
        }

    def _unit_test_skips(self) -> set:
        return set()
