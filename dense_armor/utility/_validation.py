"""Shared validation helpers.

The library reproduces the same three small checks in several modules.
This file collects them once, without changing any existing class: the
copies inside ``facto/fm.py``, ``facto/ffm.py``, ``facto/fwfm.py``,
``facto/hofm.py``, ``tree/fimt.py``, ``bandit/base.py`` and
``active/base.py`` are left where they are.

The functions here are:

- ``as_dict(x)``: return a dict from a dict or from an object with
  ``to_dict()``; raise ``TypeError`` otherwise. This is the same code as
  ``_as_dict`` in the modules above, including the message of the
  exception.
- ``is_number(v)``: true when ``v`` is not ``None``, ``float(v)`` does
  not raise and the result is finite. This is the single-value form of
  ``_finite`` (``tree/fimt.py``) and of the per-value loop in ``_nan_in``
  (``facto/*.py``).
- ``finite_row(d)``: true when every value of ``d`` passes
  ``is_number``. Same as ``_finite`` in ``tree/fimt.py`` and the
  per-row part of ``_nan_in``.

The migration of the existing classes to these helpers is a proposal at
the end of the delivery: this file is additive and changes nothing.
"""

import math
from typing import Any


def as_dict(x: Any) -> dict:
    """Return ``x`` as a dict.

    Args:
        x: a dict, or an object with a callable ``to_dict`` method.

    Returns:
        ``x`` itself when it is a dict, otherwise the result of
        ``x.to_dict()``.

    Raises:
        TypeError: when ``x`` is neither a dict nor an object with
            ``to_dict()``.

    Examples:
        >>> as_dict({"a": 1.0})
        {'a': 1.0}
        >>> class S:
        ...     def to_dict(self):
        ...         return {"b": 2.0}
        >>> as_dict(S())
        {'b': 2.0}
        >>> as_dict(3)
        Traceback (most recent call last):
        ...
        TypeError: expected a dict or an object with to_dict(), got int
    """
    if isinstance(x, dict):
        return x
    to_dict = getattr(x, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(
        f"expected a dict or an object with to_dict(), got {type(x).__name__}"
    )


def is_number(v: Any) -> bool:
    """Return True when ``v`` is a finite float.

    Args:
        v: any value.

    Returns:
        False when ``v`` is ``None``, when ``float(v)`` raises, or when
        the float is not finite. True otherwise.

    Examples:
        >>> is_number(1.0)
        True
        >>> is_number("2.5")
        True
        >>> is_number(None)
        False
        >>> is_number(float("nan"))
        False
        >>> is_number("hello")
        False
    """
    if v is None:
        return False
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fv)


def finite_row(d: dict) -> bool:
    """Return True when every value of ``d`` is a finite float.

    Args:
        d: a dict of values.

    Returns:
        True when :func:`is_number` returns True for every value.

    Examples:
        >>> finite_row({"a": 1.0, "b": 2})
        True
        >>> finite_row({"a": 1.0, "b": None})
        False
        >>> finite_row({"a": float("nan")})
        False
        >>> finite_row({})
        True
    """
    for v in d.values():
        if not is_number(v):
            return False
    return True
