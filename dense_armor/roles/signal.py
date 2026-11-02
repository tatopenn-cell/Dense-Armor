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

    A Signal is a reading of one sample: a dict ``{name: float}``, so any
    estimator that takes a dict takes it, and ``array``, the same numbers
    as a JAX vector for the code that works on vectors. Several samples
    are a :class:`SignalBatch`, which is not a dict.

    Args:
        values: JAX array of shape ``(n_channels,)``.
        names: channel names, in the same order as ``values``.
        units: channel units, one string per channel.
        t: timestamp of the sample in seconds.
        missing: optional boolean mask of the same shape as ``values``;
            ``True`` marks a channel that was not read.

    Raises:
        ValueError: if ``values`` is not 1-D (use :class:`SignalBatch` for
            ``n`` samples), or if ``names``, ``units`` or ``missing`` do
            not match the channel count.
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
        if v.ndim != 1:
            raise ValueError(
                f"values has shape {v.shape}, expected (n_channels,); "
                "use SignalBatch for n samples"
            )
        n_channels = v.shape[0]
        names = list(names)
        units = list(units)
        if len(names) != n_channels:
            raise ValueError(
                f"names has {len(names)} entries, values has {n_channels} channels"
            )
        if len(units) != n_channels:
            raise ValueError(
                f"units has {len(units)} entries, values has {n_channels} channels"
            )
        self.array: jnp.ndarray = v
        self.names: list[str] = names
        self.units: list[str] = units
        self.t: float | None = t
        mask = jnp.isnan(v) if missing is None else jnp.asarray(missing, dtype=bool)
        if mask.shape != v.shape:
            raise ValueError(
                f"missing has shape {mask.shape}, values has shape {v.shape}"
            )
        self.mask: jnp.ndarray = mask
        dict.__init__(self, {name: float(v[i]) for i, name in enumerate(names)})

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

    def stacked(self, others: Iterable["Signal"]) -> "SignalBatch":
        """Stack this Signal with others along a new batch axis.

        Args:
            others: signals with the same channel names and units.

        Returns:
            A :class:`SignalBatch` of ``n + 1`` samples. It is not a dict:
            it is for the code that works on ``array``.

        Raises:
            ValueError: if the signals differ in names or units.
        """
        group = [self, *others]
        for s in group:
            if s.names != self.names or s.units != self.units:
                raise ValueError("stacked signals must share names and units")
        return SignalBatch(
            values=jnp.stack([s.array for s in group], axis=0),
            names=self.names,
            units=self.units,
            times=[s.t for s in group],
            missing=jnp.stack([s.mask for s in group], axis=0),
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


class SignalBatch:
    """``n`` samples of ``d`` channels, for the code that works on vectors.

    This is not a dict, and it does not behave like one: an estimator with
    ``learn_one``, ``predict_one`` or ``score_one`` takes one sample, a
    :class:`Signal` or a plain dict, and cannot take a batch. A batch is
    read through ``array``, the JAX matrix of shape ``(n, d)``, by the code
    that works on whole arrays; ``row(i)`` gives the sample ``i`` as a
    :class:`Signal`.

    Args:
        values: JAX array of shape ``(n, d)``, one row per sample.
        names: channel names, one per column.
        units: channel units, one string per column.
        times: optional timestamps, one per row; ``None`` for a row whose
            time is not known.
        missing: optional boolean mask of the shape of ``values``; ``True``
            marks a channel that was not read. ``None`` marks the ``nan``.

    Attributes:
        array: the values, shape ``(n, d)``.
        mask: the missing mask, shape ``(n, d)``.
        times: the timestamps, a tuple of ``n``.
        t: the timestamp shared by every row, or ``None`` when the
            timestamps differ or are unknown.

    Raises:
        ValueError: if ``values`` is not 2-D, or if ``names``, ``units``,
            ``times`` or ``missing`` do not match its shape.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dense_armor.roles.signal import Signal
        >>> a = Signal(jnp.array([1.0, 2.0]), ["x", "y"], ["m", "m"], t=0.0)
        >>> b = Signal(jnp.array([3.0, 4.0]), ["x", "y"], ["m", "m"], t=0.1)
        >>> batch = a.stacked([b])
        >>> batch.array.shape
        (2, 2)
        >>> batch.row(1)["x"]
        3.0
    """

    def __init__(
        self,
        values: Any,
        names: Iterable[str],
        units: Iterable[str],
        times: Iterable[float | None] | None = None,
        missing: Any | None = None,
    ) -> None:
        v = jnp.asarray(values)
        if v.ndim != 2:
            raise ValueError(f"values has shape {v.shape}, expected (n, d)")
        names = list(names)
        units = list(units)
        if len(names) != v.shape[1]:
            raise ValueError(
                f"names has {len(names)} entries, values has {v.shape[1]} channels"
            )
        if len(units) != v.shape[1]:
            raise ValueError(
                f"units has {len(units)} entries, values has {v.shape[1]} channels"
            )
        ts = tuple(times) if times is not None else (None,) * v.shape[0]
        if len(ts) != v.shape[0]:
            raise ValueError(
                f"times has {len(ts)} entries, values has {v.shape[0]} rows"
            )
        mask = jnp.isnan(v) if missing is None else jnp.asarray(missing, dtype=bool)
        if mask.shape != v.shape:
            raise ValueError(
                f"missing has shape {mask.shape}, values has shape {v.shape}"
            )
        self.array: jnp.ndarray = v
        self.names: list[str] = names
        self.units: list[str] = units
        self.mask: jnp.ndarray = mask
        self.times: tuple[float | None, ...] = ts
        self.t: float | None = ts[0] if ts and all(tt == ts[0] for tt in ts) else None

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def row(self, i: int) -> Signal:
        """Sample ``i`` as a :class:`Signal`."""
        return Signal(
            values=self.array[i],
            names=self.names,
            units=self.units,
            t=self.times[i],
            missing=self.mask[i],
        )

    def __repr__(self) -> str:
        return (
            f"SignalBatch(n={len(self)}, names={self.names}, "
            f"units={self.units}, t={self.t})"
        )
