"""Bounded-buffer shuffle for streams without timestamps.

Examples:
    >>> import numpy as np
    >>> from dense_armor.utility.stream import iter_array, shuffle
    >>> X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> stream = iter_array(X, names=["x"], units=[""])
    >>> out = [float(sig["x"]) for sig, _ in shuffle(stream, buffer_size=3, seed=0)]
    >>> sorted(out)
    [1.0, 2.0, 3.0, 4.0, 5.0]
"""

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from dense_armor.roles import Signal


def shuffle(
    stream: Iterable[tuple[Signal, Any]],
    buffer_size: int,
    seed: int | None = 0,
) -> Iterator[tuple[Signal, Any]]:
    """Shuffle a stream with a bounded buffer.

    Args:
        stream: iterable of ``(Signal, y)`` pairs.
        buffer_size: maximum samples held in memory. Must be >= 1.
        seed: seed of the random generator.

    Yields:
        The same samples in a shuffled order.

    Raises:
        ValueError: if ``buffer_size < 1``.
    """
    if buffer_size < 1:
        raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
    rng = np.random.default_rng(seed)
    buf: list[tuple[Signal, Any]] = []
    for item in stream:
        if len(buf) < buffer_size:
            buf.append(item)
            continue
        j = int(rng.integers(0, buffer_size))
        yield buf[j]
        buf[j] = item
    order = rng.permutation(len(buf))
    for i in order:
        yield buf[int(i)]
