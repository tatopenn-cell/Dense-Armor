"""Stress: what do the existing pipelines do in a flash crash?

A mini flash crash is a large price move within milliseconds, followed by
a partial recovery. Braun et al. (2017) measure the events on real trade
data: the price "increases or decreases strongly within milliseconds"
(abstract, page 1), "less than 40 percent recover completely" (page 1),
and 25 percent are "almost recovered after only one trade" (page 1).

This module generates a synthetic flash crash with those numbers as
parameters and runs the existing finance pipelines on it, without
changing them. The generator is deterministic for a seed.

Reference: Braun, T., Fiegen, J. A., Wagner, D. C., Krause, S. and
Guhr, T. (2017). *Impact and Recovery Process of Mini Flash Crashes: An
Empirical Study.* arXiv:1707.05580. Duration in milliseconds, partial
recovery and the full-recovery fraction below 40 percent: page 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlashCrashResult:
    """The synthetic crash and its parameters.

    Attributes:
        n_ticks: number of ticks.
        crash_at: index of the first tick of the crash.
        duration_ticks: length of the crash in ticks.
        drop: the fractional drop of the crash.
        recovery_fraction: the fraction of the drop that recovers.
        prices: the price path.
    """

    n_ticks: int
    crash_at: int
    duration_ticks: int
    drop: float
    recovery_fraction: float
    prices: np.ndarray


def synthetic_flash_crash(
    n: int = 2000,
    seed: int = 0,
    crash_at: int = 1000,
    duration_ticks: int = 5,
    drop: float = 0.05,
    recovery_fraction: float = 0.6,
    base: float = 100.0,
    vol: float = 0.001,
) -> FlashCrashResult:
    """A synthetic flash crash with a planted drop and recovery.

    The price walks with Gaussian noise of standard deviation ``vol``.
    At ``crash_at`` it drops by ``drop`` over ``duration_ticks`` ticks,
    then recovers by ``recovery_fraction * drop`` over the same number
    of ticks, and then walks again.

    Args:
        n: number of ticks.
        seed: RNG seed.
        crash_at: index of the first tick of the crash.
        duration_ticks: length of the crash, in ticks.
        drop: the fractional drop, as a positive number.
        recovery_fraction: the fraction of the drop that recovers, in
            [0, 1].
        base: starting price.
        vol: per-tick volatility of the normal walk.

    Raises:
        ValueError: on a negative ``drop``, a ``recovery_fraction``
            outside [0, 1], or a non-positive ``duration_ticks``.
    """
    if drop < 0:
        raise ValueError(f"drop must be >= 0, got {drop}")
    if not (0.0 <= recovery_fraction <= 1.0):
        raise ValueError(
            f"recovery_fraction must be in [0, 1], got {recovery_fraction}")
    if duration_ticks < 1:
        raise ValueError(
            f"duration_ticks must be >= 1, got {duration_ticks}")
    rng = np.random.default_rng(seed)
    w = np.empty(n, dtype=float)
    w[0] = base
    for i in range(1, n):
        w[i] = w[i - 1] * (1.0 + vol * rng.normal())
    dev = np.ones(n, dtype=float)
    end_crash = min(crash_at + duration_ticks, n)
    k = end_crash - crash_at
    if k > 0:
        dev[crash_at:end_crash] = np.linspace(1.0, 1.0 - drop, k)
    end_rec = min(end_crash + duration_ticks, n)
    m = end_rec - end_crash
    final_level = 1.0 - drop * (1.0 - recovery_fraction)
    if m > 0:
        dev[end_crash:end_rec] = np.linspace(1.0 - drop, final_level, m)
    dev[end_rec:] = final_level
    p = w * dev
    return FlashCrashResult(
        n_ticks=n, crash_at=crash_at, duration_ticks=duration_ticks,
        drop=drop, recovery_fraction=recovery_fraction, prices=p)


def crash_metrics(result: FlashCrashResult) -> dict:
    """The realised drop and recovery of a generated crash."""
    p = result.prices
    i = result.crash_at
    j = min(i + result.duration_ticks - 1, len(p) - 1)
    k = min(j + result.duration_ticks, len(p) - 1)
    pre = float(p[i - 1]) if i > 0 else float(p[0])
    low = float(p[j])
    after = float(p[k])
    realised_drop = 1.0 - low / pre if pre > 0 else float("nan")
    recovered = (after - low) / (pre - low) if pre > low else 0.0
    return {
        "pre": pre, "low": low, "after": after,
        "realised_drop": realised_drop,
        "realised_recovery": float(recovered),
    }
