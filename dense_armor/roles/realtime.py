"""Real-time contract for robot estimators.

A control loop has a period. Every estimator inside the loop must run
inside that period, on every sample, or the loop misses its deadline
and the robot becomes unstable. This module gives the loop three tools:

- **declared budgets** — an estimator announces its own ``budget_s``
  (the maximum acceptable latency) and its ``memory_class``
  (``"O(1)"`` or ``"O(window)"``);
- **measured profiles** — :func:`profile` runs the estimator on a
  stream, warms up the JIT, and returns p50 / p99 / max latency and the
  memory growth over the stream;
- **a safety check at build time** — :class:`RealtimePipeline` refuses
  to start if the sum of the per-step p99 exceeds the loop period.

A separate facility is the **pure step**. An estimator that exposes
``step(state, signal) -> (state, output)`` as a pure function can be
executed inside :func:`jax.lax.scan`, which turns a whole stream into a
single fused XLA computation. The module shows one worked example, an
exponentially weighted mean.

References
----------
West, D. H. D. (1979). Updating mean and variance estimates: an
    improved method. Communications of the ACM 22(9), 532-535.
"""
import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dense_armor.roles import Root
from dense_armor.roles.signal import Signal


DEFAULT_WARMUP = 10


@dataclass
class ProfileResult:
    """Latency and memory profile of one estimator over a stream.

    Attributes:
        p50_s: median per-sample latency after warm-up, in seconds.
        p99_s: 99th-percentile per-sample latency after warm-up.
        max_s: worst per-sample latency observed after warm-up.
        memory_start_b: bytes held just after warm-up.
        memory_end_b: bytes held at the end of the stream.
        memory_growth_b: ``memory_end_b - memory_start_b``.
        memory_class: the estimator's declared class, or ``"unknown"``.
        budget_s: the estimator's declared budget, or ``None``.
        n_samples: number of samples processed after warm-up.
        n_warmup: number of samples used only for warm-up.
    """

    p50_s: float
    p99_s: float
    max_s: float
    memory_start_b: int
    memory_end_b: int
    memory_growth_b: int
    memory_class: str
    budget_s: Optional[float]
    n_samples: int
    n_warmup: int

    @property
    def fits(self) -> Optional[bool]:
        """``True`` if ``p99_s`` is at or below the declared ``budget_s``."""
        if self.budget_s is None:
            return None
        return self.p99_s <= self.budget_s


def _budget_of(est: Any) -> Optional[float]:
    return getattr(est, "budget_s", None)


def _memory_class_of(est: Any) -> str:
    return getattr(est, "memory_class", "unknown")


def profile(
    est: Any,
    signals: Sequence[Signal],
    warmup: int = DEFAULT_WARMUP,
    method: str = "learn_one",
) -> ProfileResult:
    """Measure per-sample latency and memory growth of an estimator.

    The estimator is driven one sample at a time through ``method``
    (default ``learn_one``). The first ``warmup`` samples are processed
    without timing, so a possible JAX JIT retrace does not enter the
    distribution. After that, every call is timed with
    :func:`time.perf_counter`. Memory is sampled with
    ``est._raw_memory_usage`` if the estimator exposes it; otherwise the
    growth is reported as ``0``.

    Args:
        est: the estimator to profile.
        signals: the stream of :class:`Signal` samples.
        warmup: number of leading samples to skip when timing.
        method: name of the per-sample method to call. Default
            ``"learn_one"``.

    Returns:
        A :class:`ProfileResult`. ``fits`` is ``None`` when the
        estimator does not declare a ``budget_s``.
    """
    fn: Callable = getattr(est, method)
    has_memory = hasattr(est, "_raw_memory_usage")
    if has_memory:
        gc.collect()
    for sig in signals[:warmup]:
        fn(sig)
    if has_memory:
        gc.collect()
        start_bytes = int(est._raw_memory_usage)
    else:
        start_bytes = 0
    latencies: list[float] = []
    for sig in signals[warmup:]:
        t0 = time.perf_counter()
        fn(sig)
        latencies.append(time.perf_counter() - t0)
    if has_memory:
        gc.collect()
        end_bytes = int(est._raw_memory_usage)
    else:
        end_bytes = 0
    arr = np.asarray(latencies) if latencies else np.zeros(1)
    return ProfileResult(
        p50_s=float(np.percentile(arr, 50)),
        p99_s=float(np.percentile(arr, 99)),
        max_s=float(arr.max()),
        memory_start_b=start_bytes,
        memory_end_b=end_bytes,
        memory_growth_b=end_bytes - start_bytes,
        memory_class=_memory_class_of(est),
        budget_s=_budget_of(est),
        n_samples=len(latencies),
        n_warmup=min(warmup, len(signals)),
    )


@dataclass
class RealtimePipeline:
    """A list of estimators that must fit inside a control-loop period.

    Each step declares its own ``budget_s``. The pipeline refuses to
    start if the sum of the declared budgets exceeds ``period_s``, or if
    any step does not declare one. This check happens at construction
    time, before the loop ever runs.

    Args:
        steps: the estimators, in execution order.
        period_s: the control-loop period in seconds.
        require_budget: if ``True`` (default), every step must declare a
            ``budget_s``. Set to ``False`` to bypass the check when the
            caller is happy to reason about it later.

    Raises:
        ValueError: if the sum of the declared budgets exceeds the
            period, or if ``require_budget`` is ``True`` and some step
            does not declare one.
    """

    steps: list[Any]
    period_s: float
    require_budget: bool = True
    _total_budget_s: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.period_s <= 0:
            raise ValueError(f"period_s must be > 0, got {self.period_s}")
        total = 0.0
        missing: list[int] = []
        for i, s in enumerate(self.steps):
            b = _budget_of(s)
            if b is None:
                missing.append(i)
            else:
                total += b
        if missing and self.require_budget:
            raise ValueError(
                f"steps {missing} do not declare budget_s; "
                "set require_budget=False to bypass"
            )
        if total > self.period_s:
            raise ValueError(
                f"sum of budgets {total:.6f} s exceeds period "
                f"{self.period_s:.6f} s"
            )
        self._total_budget_s = total

    @property
    def total_budget_s(self) -> Optional[float]:
        return self._total_budget_s

    @property
    def headroom_s(self) -> Optional[float]:
        if self._total_budget_s is None:
            return None
        return self.period_s - self._total_budget_s

    def learn_one(self, sig: Signal) -> list[Any]:
        """Call ``learn_one`` on every step in order, return their outputs."""
        return [step.learn_one(sig) for step in self.steps]


class PureEW(Root):
    """Exponentially weighted mean in pure form, ready for ``lax.scan``.

    The state is ``(mean, n)`` as a JAX pytree: a scalar float and an
    int. The :meth:`step` method is pure — no side effects, no state on
    ``self`` — so the whole stream can be fused into one XLA computation
    with :func:`jax.lax.scan`.

    Args:
        alpha: weight of the new sample, in ``(0, 1]``.
        budget_s: declared latency budget for :class:`RealtimePipeline`.
        memory_class: declared memory class, ``"O(1)"`` by construction.
    """

    budget_s: Optional[float] = 1e-4
    memory_class: str = "O(1)"

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def init_state(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the initial ``(mean, n)`` state."""
        return jnp.array(0.0), jnp.array(0, dtype=jnp.int32)

    def step(
        self,
        state: tuple[jnp.ndarray, jnp.ndarray],
        v: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """One pure step: ``(state, sample) -> (state, mean)``.

        Args:
            state: the ``(mean, n)`` pair.
            v: one scalar sample.

        Returns:
            The new state and the new mean.
        """
        mean, n = state
        n_new = n + 1
        first = n == 0
        new_mean = jnp.where(first, v, mean + self.alpha * (v - mean))
        return (new_mean, n_new), new_mean

    def scan(self, stream: jnp.ndarray) -> jnp.ndarray:
        """Run ``step`` over a whole stream with ``lax.scan``.

        Args:
            stream: 1-D array of samples.

        Returns:
            1-D array of running means, one per input sample.
        """
        _, means = jax.lax.scan(self.step, self.init_state(), stream)
        return means
