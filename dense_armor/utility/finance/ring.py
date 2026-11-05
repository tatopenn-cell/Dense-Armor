"""A ring buffer with preallocated storage for one producer and one consumer."""

import numpy as np


class RingBuffer:
    """A queue of fixed capacity, with the storage preallocated.

    ``push(row)`` copies ``row`` into the next free slot and returns
    ``True``, or returns ``False`` when the buffer is full and leaves it
    unchanged (it never blocks and never overwrites). ``pop_into(out)``
    copies the oldest row into ``out`` and returns ``True``, or returns
    ``False`` when the buffer is empty. The order is first in, first out.

    The class has two integer counters, ``head`` (written only by the
    producer) and ``tail`` (written only by the consumer), and takes no
    lock. Its correctness for one producer thread and one consumer thread
    rests on this rule and on the global interpreter lock of CPython.
    No claim is made for a free-threaded build of Python, nor for several
    producers or several consumers.

    Args:
        capacity: the maximum number of rows stored; at least 1.
        width: the number of float64 values in one row; at least 1.

    Raises:
        ValueError: if ``capacity`` or ``width`` is below 1.

    Examples:
        >>> b = RingBuffer(3, 2)
        >>> b.push(np.array([1.0, 2.0]))
        True
        >>> b.push(np.array([3.0, 4.0]))
        True
        >>> out = np.zeros(2)
        >>> b.pop_into(out), out.tolist()
        (True, [1.0, 2.0])
        >>> len(b)
        1

    References:
        Our design: the classic circular queue with one writer per
        index. No paper.
    """

    def __init__(self, capacity: int, width: int) -> None:
        if int(capacity) < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if int(width) < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        self.capacity = int(capacity)
        self.width = int(width)
        self._buf: np.ndarray = np.zeros(
            (self.capacity, self.width), dtype=np.float64
        )
        self.head = 0
        self.tail = 0
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def _check_row(self, row: np.ndarray) -> None:
        if row.shape != (self.width,):
            raise ValueError(f"row must have shape ({self.width},), got {row.shape}")

    def push(self, row: np.ndarray) -> bool:
        """Copy one row in, or return ``False`` if the buffer is full."""
        arr = np.asarray(row, dtype=np.float64)
        self._check_row(arr)
        if self._len == self.capacity:
            return False
        self._buf[self.head] = arr
        self.head = (self.head + 1) % self.capacity
        self._len += 1
        return True

    def pop_into(self, out: np.ndarray) -> bool:
        """Copy the oldest row into ``out``, or return ``False`` if empty."""
        arr = np.asarray(out, dtype=np.float64)
        self._check_row(arr)
        if self._len == 0:
            return False
        out[...] = self._buf[self.tail]
        self.tail = (self.tail + 1) % self.capacity
        self._len -= 1
        return True
