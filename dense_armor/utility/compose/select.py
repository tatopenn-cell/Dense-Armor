"""Keep or drop named channels of a :class:`Signal`.

Both transformers return a new :class:`Signal` with the same ``t``,
``units`` and ``missing`` for the channels they keep. When the input is
a plain dict, they return a plain dict.

Examples:
    >>> from dense_armor.roles import Signal
    >>> from dense_armor.utility.compose import Select, Discard
    >>> s = Signal(values=[1.0, 2.0, 3.0], names=["a", "b", "c"],
    ...            units=["", "", ""], t=0.5)
    >>> sorted(Select(keys=("a", "c")).transform_one(s))
    ['a', 'c']
    >>> sorted(Discard(keys=("b",)).transform_one(s))
    ['a', 'c']
"""

from typing import Any

from dense_armor.roles import Signal, Transformer


def _subset(x: Any, keep: list[str]) -> Any:
    if isinstance(x, Signal):
        idx = [x.names.index(k) for k in keep if k in x.names]
        if not idx:
            arr = x.array[:0]
            mask = x.mask[:0]
        else:
            arr = x.array[..., idx]
            mask = x.mask[..., idx]
        names = [x.names[i] for i in idx]
        units = [x.units[i] for i in idx]
        return Signal(
            values=arr,
            names=names,
            units=units,
            t=x.t,
            missing=mask,
        )
    return {k: x[k] for k in keep if k in x}


class Select(Transformer):
    """Keep only the named channels.

    Args:
        keys: names of the channels to keep, in the order given. Keys
            not present in the input are silently ignored.

    Examples:
        >>> from dense_armor.roles import Signal
        >>> from dense_armor.utility.compose import Select
        >>> s = Signal(values=[1.0, 2.0], names=["a", "b"],
        ...            units=["", ""], t=0.0)
        >>> sorted(Select(keys=("b",)).transform_one(s))
        ['b']
    """

    def __init__(self, keys: tuple[str, ...] = ()) -> None:
        self.keys = list(keys)

    def _reset(self) -> None:
        pass

    def _unit_test_skips(self) -> set:
        return set()

    def learn_one(self, x: Any, y: Any = None, t: float | None = None) -> "Select":
        """No-op: the transformer is stateless."""
        self._time_step(t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> Any:
        """Return a copy with only the selected keys.

        Args:
            x: a :class:`Signal` or a plain dict.
            t: unused.

        Returns:
            A new :class:`Signal` (or dict) with only the selected keys.
        """
        _ = t
        return _subset(x, self.keys)


class Discard(Transformer):
    """Drop the named channels.

    Args:
        keys: names of the channels to drop. Keys not present in the
            input are silently ignored.

    Examples:
        >>> from dense_armor.roles import Signal
        >>> from dense_armor.utility.compose import Discard
        >>> s = Signal(values=[1.0, 2.0], names=["a", "b"],
        ...            units=["", ""], t=0.0)
        >>> sorted(Discard(keys=("a",)).transform_one(s))
        ['b']
    """

    def __init__(self, keys: tuple[str, ...] = ()) -> None:
        self.keys = list(keys)

    def _reset(self) -> None:
        pass

    def _unit_test_skips(self) -> set:
        return set()

    def learn_one(self, x: Any, y: Any = None, t: float | None = None) -> "Discard":
        """No-op: the transformer is stateless."""
        self._time_step(t)
        return self

    def transform_one(self, x: Any, t: float | None = None) -> Any:
        """Return a copy without the dropped keys.

        Args:
            x: a :class:`Signal` or a plain dict.
            t: unused.

        Returns:
            A new :class:`Signal` (or dict) without the dropped keys.
        """
        _ = t
        if isinstance(x, Signal):
            keep = [k for k in x.names if k not in self.keys]
        else:
            keep = [k for k in x if k not in self.keys]
        return _subset(x, keep)
