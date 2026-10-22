"""Learning-rate schedules.

Three schedules, each with ``__call__(epoch)`` that returns the
learning rate at a real-valued epoch and ``multiplier(epoch)`` that
returns ``self(epoch) / self(0.0)``. The cosine annealing follows
Eqn.(5), page 4 of Loshchilov and Hutter 2017, with the restart
period growing by ``t_mult`` at every restart (Figure 1, page 2).
"""

from __future__ import annotations

import math


class ConstantLR:
    """A constant learning rate.

    Args:
        lr: the learning rate. Must be positive.

    Raises:
        ValueError: if ``lr`` is not positive.

    Examples:
        >>> from dense_armor.utility.neural.schedules import ConstantLR
        >>> ConstantLR(0.1)(0.0)
        0.1
        >>> ConstantLR(0.1)(3.5)
        0.1
    """

    def __init__(self, lr: float) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = float(lr)

    def __call__(self, epoch: float) -> float:
        """Return ``lr`` regardless of the epoch."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        return self.lr

    def multiplier(self, epoch: float) -> float:
        """Return ``self(epoch) / self(0.0)``, which is 1.0."""
        return self(epoch) / self(0.0)


class StepLR:
    """Step schedule: ``lr * gamma ** floor(epoch / step)``.

    This is a textbook step schedule; the sources are silent on it.

    Args:
        lr: the base learning rate. Must be positive.
        step: the number of epochs between two drops. Must be positive.
        gamma: the factor of each drop. Must be in ``(0, 1]``.

    Raises:
        ValueError: on a non-positive ``lr`` or ``step``, or a ``gamma``
            outside ``(0, 1]``.

    Examples:
        >>> from dense_armor.utility.neural.schedules import StepLR
        >>> StepLR(0.1, 10, 0.5)(9.99)
        0.1
        >>> StepLR(0.1, 10, 0.5)(10.0)
        0.05
        >>> StepLR(0.1, 10, 0.5)(25.0)
        0.025
    """

    def __init__(self, lr: float, step: float, gamma: float) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.lr = float(lr)
        self.step = float(step)
        self.gamma = float(gamma)

    def __call__(self, epoch: float) -> float:
        """Return the learning rate at ``epoch``."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        return self.lr * (self.gamma ** math.floor(epoch / self.step))

    def multiplier(self, epoch: float) -> float:
        """Return ``self(epoch) / self(0.0)``."""
        return self(epoch) / self(0.0)


class CosineWarmRestarts:
    """Cosine annealing with warm restarts.

    Within the run ``i``, ``lr = lr_min + (1/2)(lr_max - lr_min)
    (1 + cos(pi * t_cur / t_i))`` (Eqn.(5), page 4 of Loshchilov and
    Hutter 2017), where ``t_i = t0 * t_mult ** i`` and ``t_cur`` is
    the number of epochs since the last restart. With ``t_mult = 1``
    every run has length ``t0``; with ``t_mult > 1`` the runs grow
    (Figure 1, page 2).

    Args:
        lr_max: the learning rate at the start of each run.
        lr_min: the learning rate at the end of each run.
        t0: the length of the first run.
        t_mult: the growth factor of the run length, default 1.

    Raises:
        ValueError: on a non-positive ``t0`` or ``t_mult``, or if
            ``lr_min > lr_max``.

    Examples:
        >>> from dense_armor.utility.neural.schedules import CosineWarmRestarts
        >>> c = CosineWarmRestarts(0.1, 0.0, 10)
        >>> c(0.0)
        0.1
        >>> round(c(5.0), 4)
        0.05
        >>> c(10.0)
        0.1
    """

    def __init__(
        self, lr_max: float, lr_min: float, t0: float, t_mult: float = 1.0,
    ) -> None:
        if t0 <= 0:
            raise ValueError(f"t0 must be positive, got {t0}")
        if t_mult <= 0:
            raise ValueError(f"t_mult must be positive, got {t_mult}")
        if lr_min > lr_max:
            raise ValueError(
                f"lr_min {lr_min} is above lr_max {lr_max}"
            )
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.t0 = float(t0)
        self.t_mult = float(t_mult)

    def _run(self, epoch: float) -> tuple[float, float]:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        if self.t_mult == 1.0:
            i = int(epoch // self.t0)
            return epoch - i * self.t0, self.t0
        acc = 0.0
        t_i = self.t0
        while True:
            if epoch < acc + t_i:
                return epoch - acc, t_i
            acc += t_i
            t_i *= self.t_mult

    def __call__(self, epoch: float) -> float:
        """Return the learning rate at ``epoch``."""
        t_cur, t_i = self._run(epoch)
        cos_term = math.cos(math.pi * t_cur / t_i)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cos_term)

    def multiplier(self, epoch: float) -> float:
        """Return ``self(epoch) / self(0.0)``."""
        return self(epoch) / self(0.0)
