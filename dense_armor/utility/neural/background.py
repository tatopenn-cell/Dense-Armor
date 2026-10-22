"""A background learner: the main program is never blocked.

``BackgroundLearner`` wraps an estimator with ``learn_one(x, y, t=None)``
and ``predict_one(x, t=None)`` and moves the learning to a daemon
worker thread. The main thread puts the samples in a
``queue.Queue`` of ``maxsize`` and returns at once; when the queue is
full, the oldest sample is removed and counted in ``n_dropped_`` (our
choice: the main program is never blocked and the newest data are
kept). After every ``sync_every`` updates the worker makes
``served_ = copy.deepcopy(estimator)`` in one statement, and
``predict_one`` reads ``served_`` without waiting for the worker.
"""

from __future__ import annotations

import copy
import queue
import threading
from typing import Any


class _Flush:
    """A flush request: the worker sets the event when the queue is empty."""

    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()


class _Stop:
    """A stop request: the worker exits the loop."""

    __slots__ = ()


class BackgroundLearner:
    """Learn on a worker thread; predict on the last served copy.

    The worker is a daemon thread started by the first ``learn_one``.
    ``learn_one`` puts the sample in a ``queue.Queue`` of ``maxsize``
    and returns ``self`` at once; when the queue is full the oldest
    sample is removed and counted in ``n_dropped_``. After every
    ``sync_every`` updates the worker assigns
    ``served_ = copy.deepcopy(estimator)`` in one statement.
    ``predict_one`` reads ``served_`` and never waits for the worker;
    before the first copy it uses the estimator as it is.
    ``flush(timeout=None)`` waits until the queue is empty and the last
    copy is done. ``close()`` stops the thread and joins it. The class
    is a context manager. An exception raised in the worker is kept in
    ``error_`` and raised by the next call of ``learn_one``,
    ``predict_one`` or ``flush`` (then cleared).

    Args:
        estimator: an object with ``learn_one(x, y, t=None)`` and
            ``predict_one(x, t=None)``.
        maxsize: the queue length; the oldest sample is dropped when
            the queue is full. Default 64.
        sync_every: how many updates between two ``served_`` copies.
            Default 1.

    Raises:
        ValueError: if ``maxsize < 1`` or ``sync_every < 1``.

    Examples:
        >>> from dense_armor.utility.neural.background import BackgroundLearner
        >>> class Counter:
        ...     def __init__(self):
        ...         self.n = 0
        ...     def learn_one(self, x, y, t=None):
        ...         self.n += 1
        ...         return self
        ...     def predict_one(self, x, t=None):
        ...         return self.n
        >>> b = BackgroundLearner(Counter())
        >>> _ = b.learn_one({"a": 1.0}, 0.0)
        >>> _ = b.flush()
        >>> b.predict_one({"a": 1.0})
        1
        >>> b.close()
    """

    def __init__(
        self,
        estimator: Any,
        maxsize: int = 64,
        sync_every: int = 1,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be at least 1, got {maxsize}")
        if sync_every < 1:
            raise ValueError(f"sync_every must be at least 1, got {sync_every}")
        self.estimator = estimator
        self.maxsize = maxsize
        self.sync_every = sync_every
        self.served_ = copy.deepcopy(estimator)
        self.n_sent_ = 0
        self.n_learned_ = 0
        self.n_dropped_ = 0
        self.error_: BaseException | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"BackgroundLearner(n_sent_={self.n_sent_}, "
            f"n_learned_={self.n_learned_}, "
            f"n_dropped_={self.n_dropped_})"
        )

    def _check_error(self) -> None:
        with self._lock:
            e = self.error_
            if e is not None:
                self.error_ = None
                raise e

    def _start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run, name="BackgroundLearner", daemon=True
            )
            self._thread.start()

    def _sync(self) -> None:
        self.served_ = copy.deepcopy(self.estimator)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if isinstance(item, _Stop):
                    break
                if isinstance(item, _Flush):
                    self._sync()
                    item.event.set()
                    continue
                x, y, t = item
                try:
                    if t is None:
                        self.estimator.learn_one(x, y)
                    else:
                        self.estimator.learn_one(x, y, t=t)
                except BaseException as e:  # noqa: BLE001
                    with self._lock:
                        self.error_ = e
                else:
                    with self._lock:
                        self.n_learned_ += 1
                        n = self.n_learned_
                    if n % self.sync_every == 0:
                        self._sync()
            finally:
                self._queue.task_done()

    def learn_one(
        self, x: Any, y: Any = None, t: float | None = None
    ) -> BackgroundLearner:
        """Put the sample in the queue; never blocks the caller."""
        self._check_error()
        self._start()
        while True:
            try:
                self._queue.put_nowait((x, y, t))
                with self._lock:
                    self.n_sent_ += 1
                return self
            except queue.Full:
                try:
                    old = self._queue.get_nowait()
                except queue.Empty:
                    continue
                if isinstance(old, tuple):
                    with self._lock:
                        self.n_dropped_ += 1
                else:
                    self._queue.put_nowait(old)
                    continue
                self._queue.task_done()

    def predict_one(self, x: Any, t: float | None = None) -> Any:
        """Read ``served_``; never waits for the worker."""
        self._check_error()
        if t is None:
            return self.served_.predict_one(x)
        return self.served_.predict_one(x, t=t)

    def flush(self, timeout: float | None = None) -> BackgroundLearner:
        """Wait until the queue is empty and the last copy is done."""
        self._check_error()
        if self._thread is None or not self._thread.is_alive():
            self._sync()
            return self
        ev = _Flush()
        self._queue.put(ev)
        if not ev.event.wait(timeout):
            raise TimeoutError("flush timed out")
        self._check_error()
        return self

    def close(self) -> None:
        """Stop the worker and join it; a second call is a no-op."""
        if self._thread is None:
            return
        self._queue.put(_Stop())
        self._thread.join()
        self._thread = None

    def __enter__(self) -> BackgroundLearner:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
