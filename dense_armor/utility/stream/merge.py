"""Merge several streams into one, in global time order.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.stream import iter_array, merge_by_time
    >>> a = iter_array(np.array([1.0, 3.0]), t=[0.0, 2.0], names=["a"], units=[""])
    >>> b = iter_array(np.array([2.0, 4.0]), t=[1.0, 3.0], names=["b"], units=[""])
    >>> for sig, y in merge_by_time(a, b):
    ...     print(sig.t, list(sig.to_dict()))
    0.0 ['a']
    1.0 ['b']
    2.0 ['a']
    3.0 ['b']
"""

import heapq
from collections.abc import Iterable, Iterator
from typing import Any

from dense_armor.roles import Signal


def merge_by_time(
    *streams: Iterable[tuple[Signal, Any]],
) -> Iterator[tuple[Signal, Any]]:
    """Merge streams in time order.

    Args:
        *streams: iterables of ``(Signal, y)`` pairs, each already in
            non-decreasing order of ``Signal.t``.

    Yields:
        ``(signal, y)`` pairs, in non-decreasing order of ``t``. Equal
        timestamps keep the order of the arguments.

    Raises:
        ValueError: if a sample has ``t=None``.
    """
    heap: list[tuple[float, int, Signal, Any, Iterator]] = []
    for i, s in enumerate(streams):
        it = iter(s)
        try:
            sig, y = next(it)
        except StopIteration:
            continue
        if sig.t is None:
            raise ValueError("merge_by_time needs a timestamp on every sample")
        heapq.heappush(heap, (float(sig.t), i, sig, y, it))
    while heap:
        _, i, sig, y, it = heapq.heappop(heap)
        yield sig, y
        try:
            sig2, y2 = next(it)
        except StopIteration:
            continue
        if sig2.t is None:
            raise ValueError("merge_by_time needs a timestamp on every sample")
        heapq.heappush(heap, (float(sig2.t), i, sig2, y2, it))
