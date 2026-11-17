"""A data stream that serves batches without loading the dataset.

``DataStream`` reads ``(x, y)`` pairs from a pair of arrays, from a pair of
``.npy`` files read through a memory map, or from a generator, and serves
them in batches. A bounded shuffle buffer mixes the order without holding
more than ``shuffle_buffer`` samples (the buffer method of the data
pipelines of the field: fill the buffer, then at each new sample emit a
random one from the buffer and put the new sample in its place; at the end
of the epoch the buffer is emptied in random order).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import jax
import numpy as np

_CHUNK = 4096


def _as_array(a: Any) -> Any:
    """A path is opened as a read-only memory map; anything else is kept."""
    if isinstance(a, (str, Path)):
        return np.load(a, mmap_mode="r")
    return a


class DataStream:
    """Batches of ``(x, y)`` from arrays, memory-mapped files or a generator.

    Each call of ``iter`` is one epoch: every sample of the source is served
    exactly once, in batches of ``batch_size``; the last batch is smaller
    when the number of samples is not a multiple of ``batch_size``.

    Args:
        source: one of
            - a pair ``(X, y)`` of arrays or of paths to ``.npy`` files (a
              path is opened with ``mmap_mode="r"``, so only the batch being
              served is read);
            - a function without arguments that returns a new iterator of
              ``(x, y)`` samples, called once per epoch.
        batch_size: the number of samples of a batch, at least 1.
        shuffle_buffer: the size of the shuffle buffer in samples; 0 serves
            the samples in the order of the source.
        key: a ``jax.random`` key. Required when ``shuffle_buffer > 0``; the
            order of epoch ``e`` is drawn from ``fold_in(key, e)``.

    Raises:
        ValueError: on ``batch_size < 1``, a negative ``shuffle_buffer``, a
            ``shuffle_buffer`` without ``key``, an ``X`` and a ``y`` of
            different length, or a source that is neither a pair nor a
            function.

    Examples:
        >>> import numpy as np
        >>> X = np.arange(10, dtype=np.float32).reshape(5, 2)
        >>> y = np.arange(5)
        >>> s = DataStream((X, y), batch_size=2)
        >>> [len(b[1]) for b in s]
        [2, 2, 1]
    """

    def __init__(
        self,
        source: Any,
        batch_size: int,
        shuffle_buffer: int = 0,
        key: Any = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if shuffle_buffer < 0:
            raise ValueError(f"shuffle_buffer must be >= 0, got {shuffle_buffer}")
        if shuffle_buffer > 0 and key is None:
            raise ValueError("shuffle_buffer > 0 requires a jax.random key")
        self.batch_size = int(batch_size)
        self.shuffle_buffer = int(shuffle_buffer)
        self.key = key
        self.epoch_ = 0
        self._factory: Callable[[], Iterable[Any]] | None = None
        self.X: Any = None
        self.y: Any = None
        if callable(source):
            self._factory = source
        elif isinstance(source, tuple) and len(source) == 2:
            self.X = _as_array(source[0])
            self.y = _as_array(source[1])
            if len(self.X) != len(self.y):
                raise ValueError(
                    f"X and y have different lengths: {len(self.X)} and {len(self.y)}"
                )
        else:
            raise ValueError(
                "source must be a pair (X, y) or a function returning an iterator"
            )

    def __len__(self) -> int:
        """The number of batches of an epoch, for an array source."""
        if self.X is None:
            raise TypeError("the length of a generator stream is not known")
        return -(-len(self.X) // self.batch_size)

    @property
    def n_samples(self) -> int | None:
        """The number of samples of an array source, ``None`` for a generator."""
        return None if self.X is None else len(self.X)

    def _samples(self) -> Iterator[tuple[Any, Any]]:
        if self._factory is not None:
            it = iter(self._factory())
            for x, y in it:
                yield np.asarray(x), np.asarray(y)
            return
        n = len(self.X)
        for start in range(0, n, _CHUNK):
            xs = np.asarray(self.X[start : start + _CHUNK])
            ys = np.asarray(self.y[start : start + _CHUNK])
            for i in range(len(xs)):
                yield xs[i], ys[i]

    def _batches_in_order(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(self.X)
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            yield np.asarray(self.X[start:stop]), np.asarray(self.y[start:stop])

    def _batches_shuffled(self, rng: np.random.Generator):
        size = self.shuffle_buffer
        buf_x: list[Any] = []
        buf_y: list[Any] = []
        out_x: list[Any] = []
        out_y: list[Any] = []

        def emit(x: Any, y: Any):
            out_x.append(x)
            out_y.append(y)
            if len(out_x) == self.batch_size:
                batch = (np.stack(out_x), np.stack(out_y))
                out_x.clear()
                out_y.clear()
                return batch
            return None

        for x, y in self._samples():
            if len(buf_x) < size:
                buf_x.append(x)
                buf_y.append(y)
                continue
            j = int(rng.integers(size))
            batch = emit(buf_x[j], buf_y[j])
            buf_x[j], buf_y[j] = x, y
            if batch is not None:
                yield batch
        for j in rng.permutation(len(buf_x)):
            batch = emit(buf_x[int(j)], buf_y[int(j)])
            if batch is not None:
                yield batch
        if out_x:
            yield np.stack(out_x), np.stack(out_y)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        epoch = self.epoch_
        self.epoch_ += 1
        if self.shuffle_buffer == 0:
            if self.X is not None:
                return self._batches_in_order()
            return self._batches_of_a_generator()
        sub = jax.random.fold_in(self.key, epoch)
        seed = int(jax.random.randint(sub, (), 0, 2**31 - 1))
        return self._batches_shuffled(np.random.default_rng(seed))

    def _batches_of_a_generator(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        out_x: list[Any] = []
        out_y: list[Any] = []
        for x, y in self._samples():
            out_x.append(x)
            out_y.append(y)
            if len(out_x) == self.batch_size:
                yield np.stack(out_x), np.stack(out_y)
                out_x.clear()
                out_y.clear()
        if out_x:
            yield np.stack(out_x), np.stack(out_y)


__all__ = ["DataStream"]
