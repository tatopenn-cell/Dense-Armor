"""Typed multi-channel samples with units and a timestamp.

A :class:`Signal` is a dict of one-sample readings *plus* the array form
of the same values, the channel names, the units, and a timestamp. It
behaves as a dict, so any estimator built on :mod:`dense_armor.roles`
accepts it wherever a dict is expected:

    est.learn_one(sig, y)

At the same time, the vector form ``sig.array`` is a JAX array, ready
for a jitted step or a ``lax.scan``. Missing channels are stored as
``nan`` and flagged in ``sig.mask`` (``True`` where the channel is
missing); the dict form never silently replaces a missing channel with
zero.

Example:
    >>> from dense_armor.roles import Signal
    >>> import jax.numpy as jnp
    >>> s = Signal(
    ...     values=jnp.array([0.1, 0.2]),
    ...     names=["q0", "q1"],
    ...     units=["rad", "rad"],
    ...     t=0.0,
    ... )
    >>> dict(s) == {"q0": 0.1, "q1": 0.2}
    True
    >>> s["q0"]
    0.1
"""

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
import numpy as np


class Signal(dict):
    """Dict-like one-sample reading, with vector form and metadata.

    Args:
        values: JAX array of shape ``(n_channels,)`` or ``(n, d)`` for a
            batch. The first axis is the channel axis for a single
            sample, and the second axis for a batch of ``n`` samples of
            ``d`` channels.
        names: channel names, in the same order as ``values``.
        units: channel units, one string per channel.
        t: timestamp of the sample in seconds.
        missing: optional boolean mask of the same shape as ``values``;
            ``True`` marks a channel that was not read.

    Raises:
        ValueError: if ``names``, ``units`` or ``missing`` do not match
            the channel count.
    """

    def __init__(
        self,
        values: Any,
        names: Iterable[str],
        units: Iterable[str],
        t: float | None = None,
        missing: Any | None = None,
    ) -> None:
        v = jnp.asarray(values)
        n_channels = v.shape[1] if v.ndim == 2 else v.shape[0]
        names = list(names)
        units = list(units)
        if len(names) != n_channels:
            raise ValueError(
                f"names has {len(names)} entries, values has "
                f"{n_channels} channels"
            )
        if len(units) != n_channels:
            raise ValueError(
                f"units has {len(units)} entries, values has "
                f"{n_channels} channels"
            )
        self.array: jnp.ndarray = v
        self.names: list[str] = names
        self.units: list[str] = units
        self.t: float | None = t
        mask = (
            jnp.isnan(v)
            if missing is None
            else jnp.asarray(missing, dtype=bool)
        )
        if mask.shape != v.shape:
            raise ValueError(
                f"missing has shape {mask.shape}, values has shape {v.shape}"
            )
        self.mask: jnp.ndarray = mask
        d: dict[str, Any] = {}
        if v.ndim == 1:
            for i, name in enumerate(names):
                d[name] = float(v[i])
        else:
            for j, name in enumerate(names):
                d[name] = tuple(
                    float(v[i, j]) for i in range(v.shape[0])
                )
        dict.__init__(self, d)

    @classmethod
    def from_dict(
        cls,
        x: dict,
        units: dict | None = None,
        t: float | None = None,
        names: Iterable[str] | None = None,
    ) -> "Signal":
        """Build a Signal from a plain dict of scalars.

        Args:
            x: the feature dict. ``None`` values are stored as ``nan``
                and flagged in the mask.
            units: optional ``{name: unit}`` mapping. Names not present
                use ``""``.
            t: timestamp.
            names: explicit channel order. ``None`` uses the sorted keys
                so the result is deterministic.
        """
        names = list(names) if names is not None else sorted(x)
        vals: list[float] = []
        mask: list[bool] = []
        for name in names:
            v = x.get(name, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(float("nan"))
                mask.append(True)
            else:
                vals.append(float(v))
                mask.append(False)
        units_list = [(units or {}).get(name, "") for name in names]
        return cls(
            values=jnp.asarray(vals),
            names=names,
            units=units_list,
            t=t,
            missing=jnp.asarray(mask),
        )

    def to_dict(self) -> dict:
        """Return the plain ``{name: value}`` dict."""
        return dict(self)

    def stacked(self, others: Iterable["Signal"]) -> "Signal":
        """Stack this Signal with others along a new batch axis.

        Args:
            others: signals with the same channel names and units.

        Returns:
            A new Signal with ``array`` of shape ``(n + 1, d)``.
        """
        group = [self, *others]
        for s in group:
            if s.names != self.names or s.units != self.units:
                raise ValueError("stacked signals must share names and units")
        v = jnp.stack([s.array for s in group], axis=0)
        m = jnp.stack([s.mask for s in group], axis=0)
        ts = [s.t for s in group]
        t0 = ts[0] if all(tt == ts[0] for tt in ts) else None
        return Signal(
            values=v,
            names=self.names,
            units=self.units,
            t=t0,
            missing=m,
        )

    @property
    def n_missing(self) -> int:
        """Number of missing channels in this sample."""
        return int(jnp.sum(self.mask))

    def __repr__(self) -> str:
        head = f"Signal(names={self.names}, units={self.units}, t={self.t}"
        if self.n_missing:
            head += f", n_missing={self.n_missing}"
        return head + ")"
