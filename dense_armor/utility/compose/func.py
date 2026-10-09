"""Apply a user function to channels of a :class:`Signal`.

By default the function is called once per channel with the value of
that channel. When ``keys`` is empty and ``whole`` is False, every
channel is transformed. When ``whole`` is True, the function is called
once with the whole dict and must return a dict. A ``None`` result
skips the channel.

Examples:
    >>> from dense_armor.utility.compose import FuncTransformer
    >>> f = FuncTransformer(lambda v: v * 2.0, keys=("a",))
    >>> f.transform_one({"a": 1.5, "b": 3.0})
    {'a': 3.0, 'b': 3.0}
    >>> from dense_armor.roles import Signal
    >>> s = Signal(values=[1.0, 2.0], names=["a", "b"], units=["", ""], t=0.0)
    >>> sorted(FuncTransformer(lambda v: v + 1.0).transform_one(s))
    ['a', 'b']
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from dense_armor.roles import Signal, Transformer


class FuncTransformer(Transformer):
    """Apply ``fn`` to channels, or to the whole dict.

    Args:
        fn: a callable taking one value (when ``whole`` is False) or
            the whole dict (when ``whole`` is True), and returning the
            new value or ``None`` to skip the channel.
        keys: names of the channels to transform. ``None`` transforms
            every channel when ``whole`` is False.
        whole: if True, ``fn`` receives the whole dict and must return
            a dict, which replaces the output.

    Examples:
        >>> from dense_armor.utility.compose import FuncTransformer
        >>> f = FuncTransformer(
        ...     lambda d: {"y": sum(d.values())}, whole=True
        ... )
        >>> f.transform_one({"a": 1.0, "b": 2.0})
        {'y': 3.0}
    """

    def __init__(
        self,
        fn: Callable,
        keys: tuple[str, ...] | None = None,
        whole: bool = False,
    ) -> None:
        self.fn = fn
        self.keys = None if keys is None else list(keys)
        self.whole = bool(whole)

    def _reset(self) -> None:
        pass

    def _unit_test_skips(self) -> set:
        return set()

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> "FuncTransformer":
        """No-op: the transformer is stateless."""
        self._time_step(t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> Any:
        """Apply the function and return the new sample.

        Args:
            x: a :class:`Signal` or a plain dict.
            t: unused.

        Returns:
            A new :class:`Signal` (or dict) with the transformed values.
        """
        _ = t
        if self.whole:
            return self._transform_whole(x)
        return self._transform_channels(x)

    def _transform_whole(self, x: Any) -> Any:
        out = self.fn(dict(x))
        if isinstance(x, Signal):
            vals = [out[k] for k in out]
            names = list(out.keys())
            units = ["" for _ in names]
            return Signal(
                values=np.asarray(vals, dtype=float),
                names=names,
                units=units,
                t=x.t,
            )
        return dict(out)

    def _transform_channels(self, x: Any) -> Any:
        if isinstance(x, Signal):
            vals = [float(v) for v in x.array]
            targets = self.keys if self.keys is not None else list(x.names)
            for k in targets:
                if k not in x.names:
                    continue
                j = x.names.index(k)
                new = self.fn(vals[j])
                if new is not None:
                    vals[j] = float(new)
            return Signal(
                values=np.asarray(vals, dtype=float),
                names=list(x.names),
                units=list(x.units),
                t=x.t,
                missing=x.mask,
            )
        out = dict(x)
        targets = self.keys if self.keys is not None else list(out.keys())
        for k in targets:
            if k not in out:
                continue
            new = self.fn(out[k])
            if new is not None:
                out[k] = new
        return out
