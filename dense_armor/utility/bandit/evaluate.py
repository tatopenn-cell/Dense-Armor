"""Offline replay evaluation of a policy on a stream of logged events.

The estimator is the one of Li et al. (2011): walk the logged stream,
keep the events whose arm matches the arm the policy would have chosen
given the same context, and average the rewards of the kept events. With
a uniformly random logging policy every event is kept with probability
``1 / K`` independent of the others, so the average of the kept rewards
is an unbiased estimate of the policy's expected reward.
"""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

Event = tuple[dict, Any, float]


def offline_replay(
    stream: Iterable[Event],
    policy: Callable[[dict], Any],
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Estimate the expected reward of ``policy`` from a logged stream.

    The estimator is the one of Algorithm 2 of Li et al. (2011): keep the
    events where ``policy(context) == arm_logged``, then average the
    rewards of the kept events. The interval is a bootstrap percentile
    interval, an addition to the paper that the paper does not fix; it is
    the choice of this implementation.

    Args:
        stream: an iterable of ``(context, arm, reward)`` triples, all
            logged by a uniformly random policy.
        policy: a callable ``policy(context) -> arm``.
        n_bootstrap: number of bootstrap resamples for the interval.
        alpha: two-sided level of the interval, ``0.05`` for 95 %.
        seed: seed of the random generator for the bootstrap.

    Returns:
        A dict with keys ``mean`` (the estimate), ``ci_low``, ``ci_high``
        (the bootstrap interval), ``n_retained`` (number of kept events)
        and ``n_total`` (number of events read from the stream).

    Raises:
        ValueError: if the stream is empty, if no event is retained, or
            if ``alpha`` is outside ``(0, 1)`` or ``n_bootstrap < 1``.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.bandit.evaluate import offline_replay
        >>> rng = np.random.default_rng(0)
        >>> events = []
        >>> for _ in range(2000):
        ...     ctx = {"x": float(rng.uniform())}
        ...     arm = int(rng.integers(2))
        ...     r = float(rng.binomial(1, [0.8, 0.2][arm]))
        ...     events.append((ctx, arm, r))
        >>> out = offline_replay(events, lambda c: 0, seed=0)
        >>> round(out["mean"], 2)
        0.79
        >>> out["ci_low"] < 0.8 < out["ci_high"]
        True

    References:
        Li, L., Chu, W., Langford, J., Wang, X. (2011). Unbiased offline
            evaluation of contextual-bandit-based news article
            recommendation algorithms. In WSDM 2011. Algorithm 1 and
            Theorem 1, page 4: the rejection-sampling estimator is
            unbiased when the logging policy is uniformly random.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be at least 1")
    kept: list[float] = []
    n_total = 0
    for ctx, arm, reward in stream:
        n_total += 1
        if policy(ctx) == arm:
            try:
                r = float(reward)
            except (TypeError, ValueError):
                continue
            if np.isfinite(r):
                kept.append(r)
    if n_total == 0:
        raise ValueError("the stream is empty")
    if not kept:
        raise ValueError("no event was retained by the policy")
    arr = np.asarray(kept, dtype=float)
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    n = len(arr)
    samples = np.empty(int(n_bootstrap))
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        samples[i] = arr[idx].mean()
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "n_retained": n,
        "n_total": n_total,
    }
