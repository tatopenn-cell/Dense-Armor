"""Mini-batch generator.

``minibatches(n, batch_size, key=None, drop_last=False)`` yields index
arrays of shape ``(batch_size,)`` (the last one shorter unless
``drop_last=True``) that cover ``range(n)`` once. With ``key=None``
the order is ``0, 1, ..., n-1``; with a ``jax.random`` key the order
is ``jax.random.permutation(key, n)``. This is the textbook mini-batch
loop, no citation needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def minibatches(
    n: int,
    batch_size: int,
    key: Any = None,
    drop_last: bool = False,
) -> Iterator[jax.Array]:
    """Yield index arrays that cover ``range(n)`` once.

    Args:
        n: the number of items to cover.
        batch_size: the size of each batch.
        key: an optional ``jax.random`` key; without it the order is
            ``0, 1, ..., n-1``, with it the order is a permutation.
        drop_last: if True, a last batch shorter than ``batch_size``
            is not yielded.

    Yields:
        ``jax.numpy`` integer arrays, each of length ``batch_size``
        except possibly the last one.

    Raises:
        ValueError: if ``n < 1`` or ``batch_size < 1``.

    Examples:
        >>> from dense_armor.utility.neural.batching import minibatches
        >>> [b.tolist() for b in minibatches(10, 3)]
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if key is None:
        order = np.arange(n, dtype=np.int32)
    else:
        order = np.asarray(jax.random.permutation(key, n), dtype=np.int32)
    for start in range(0, n, batch_size):
        chunk = order[start : start + batch_size]
        if drop_last and len(chunk) < batch_size:
            return
        yield jnp.asarray(chunk)
