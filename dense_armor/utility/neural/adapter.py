"""A layer seen as a ``Transformer`` of the library.

The weights are not updated by ``learn_one``: a layer learns through
the ``Param.update`` calls of the training loop of N04 and N05, not
through the streaming interface. ``AsTransformer`` is the bridge that
lets a layer sit inside a pipeline of ``dense_armor.roles`` and receive
one dict at a time.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from dense_armor.roles import Transformer


class AsTransformer(Transformer):
    """A layer exposed as a ``Transformer``.

    ``transform_one`` reads ``input_keys`` from the dict in the order
    given, builds a ``float32`` vector, runs ``layer.forward(vector,
    training=False)`` and returns ``{prefix + i: float}`` per output
    element. A missing key, ``None``, NaN or a non-numeric value
    returns ``{}`` and increments ``n_missing``.

    ``budget_s = 0.015`` is ten times the p99 of ``transform_one``
    measured on this Colab runtime (CPU, 2 cores, 1.6 GHz) with a
    ``Dense(3, 2)`` layer and three input keys; it is never below the
    floor ``1e-3`` that ``check_estimator`` requires. ``memory_class``
    is ``"O(1)"``: the adapter stores only the input keys and the
    layer, and the layer's own state does not grow with the stream.

    Args:
        layer: any object with ``forward(x, training=True)``.
        input_keys: the dict keys read, in order, into the input vector.
        prefix: prefix of the output keys, default ``"z"``.

    Examples:
        >>> import jax
        >>> from dense_armor.utility.neural.adapter import AsTransformer
        >>> from dense_armor.utility.neural.layers import Dense
        >>> a = AsTransformer(Dense(3, 2, jax.random.PRNGKey(0)), ["a", "b", "c"])
        >>> sorted(a.transform_one({"a": 1.0, "b": 0.0, "c": 0.0}))
        ['z0', 'z1']
    """

    budget_s = 0.015
    memory_class = "O(1)"

    def __init__(
        self,
        layer: Any,
        input_keys: list[str],
        prefix: str = "z",
    ) -> None:
        self.layer = layer
        self.input_keys = tuple(input_keys)
        self.prefix = prefix
        self.n_missing = 0

    def __repr__(self) -> str:
        return (
            f"AsTransformer(layer={self.layer!r}, "
            f"input_keys={self.input_keys!r}, prefix={self.prefix!r})"
        )

    def _read(self, x: dict) -> list[float] | None:
        vals: list[float] = []
        for k in self.input_keys:
            if not isinstance(x, dict) or k not in x:
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
        return vals

    def transform_one(self, x: dict, t: float | None = None) -> dict:
        del t
        vals = self._read(x)
        if vals is None:
            self.n_missing += 1
            return {}
        v = jnp.asarray(vals, dtype=jnp.float32)
        y = self.layer.forward(v, training=False)
        return {f"{self.prefix}{i}": float(y[i]) for i in range(y.shape[0])}

    def learn_one(
        self, x: dict, y: Any = None, t: float | None = None
    ) -> AsTransformer:
        del x, y, t
        return self

    @classmethod
    def _unit_test_params(cls):
        import jax

        from dense_armor.utility.neural.layers import Dense

        yield {
            "layer": Dense(3, 2, jax.random.PRNGKey(0)),
            "input_keys": ("a", "b", "c"),
            "prefix": "z",
        }

    def _unit_test_skips(self) -> set:
        return set()
