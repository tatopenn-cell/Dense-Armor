"""Learning-rate schedules with warmup.

Two schedules, each with ``__call__(epoch)`` that returns the learning
rate at a real-valued epoch and ``multiplier(epoch)`` that returns
``self(epoch) / self(0.0)`` (or ``0.0`` when ``self(0.0) == 0.0``),
in the style of :mod:`dense_armor.utility.neural.schedules`.
"""

from __future__ import annotations


class LinearWarmup:
    """Linear warmup: ``lr_start`` then a linear ramp to ``lr_end``.

    ``lr(e) = lr_start + (lr_end - lr_start) * min(e / warmup, 1.0)``.

    The schedule starts at ``lr_start`` and grows linearly to
    ``lr_end`` over the first ``warmup`` epochs, then stays at
    ``lr_end``.

    Args:
        lr_start: the learning rate at epoch 0. Must be positive.
        lr_end: the learning rate at the end of the warmup. Must be
            positive and not below ``lr_start``.
        warmup: the length of the warmup in epochs. Must be positive.

    Raises:
        ValueError: on a non-positive ``lr_start``, ``lr_end`` or
            ``warmup``, or if ``lr_start > lr_end``.

    References:
        Goyal, P., Dollar, P., Girshick, R., Noordhuis, P.,
        Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., He, K.
        (2017). Accurate, large minibatch SGD. arXiv:1706.02677.
        Page 2, the warmup strategy; page 3, the linear scaling
        rule and the warmup together.

    Examples:
        >>> from dense_armor.utility.neural.schedules_more import LinearWarmup
        >>> s = LinearWarmup(0.1, 3.2, 5)
        >>> round(s(0.0), 2)
        0.1
        >>> round(s(2.5), 2)
        1.65
        >>> round(s(5.0), 2)
        3.2
        >>> round(s(7.0), 2)
        3.2
    """

    def __init__(self, lr_start: float, lr_end: float, warmup: float) -> None:
        if lr_start <= 0:
            raise ValueError(f"lr_start must be positive, got {lr_start}")
        if lr_end <= 0:
            raise ValueError(f"lr_end must be positive, got {lr_end}")
        if warmup <= 0:
            raise ValueError(f"warmup must be positive, got {warmup}")
        if lr_start > lr_end:
            raise ValueError(f"lr_start {lr_start} is above lr_end {lr_end}")
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_end)
        self.warmup = float(warmup)

    def __call__(self, epoch: float) -> float:
        """Return the learning rate at ``epoch``."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        frac = min(epoch / self.warmup, 1.0)
        return self.lr_start + (self.lr_end - self.lr_start) * frac

    def multiplier(self, epoch: float) -> float:
        """Return ``self(epoch) / self(0.0)``; 0 if ``self(0.0) == 0``."""
        base = self(0.0)
        if base == 0.0:
            return 0.0
        return self(epoch) / base


class NoamSchedule:
    """Noam schedule: linear warmup then inverse-square-root decay.

    ``lr(e) = scale * d_model^(-0.5) * min(e^(-0.5), e * warmup^(-1.5))``,
    with ``lr(0) = 0``. This is Eq.(3), page 7 of Vaswani et al. (2017):
    the learning rate grows linearly for the first ``warmup`` steps and
    then decays as the inverse square root of the step.

    Args:
        d_model: the model dimension. Must be positive.
        warmup: the number of warmup steps. Must be positive.
        scale: an overall multiplier. Must be positive, default 1.

    Raises:
        ValueError: on a non-positive ``d_model``, ``warmup`` or
            ``scale``.

    References:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
        Gomez, A. N., Kaiser, L., Polosukhin, I. (2017). Attention Is
        All You Need. arXiv:1706.03762. Section 5.3, Eq.(3), page 7:
        ``lrate = d_model^(-0.5) * min(step^(-0.5), step *
        warmup_steps^(-1.5))``; ``warmup_steps = 4000`` in the paper's
        experiments.

    Examples:
        >>> from dense_armor.utility.neural.schedules_more import NoamSchedule
        >>> s = NoamSchedule(512, 4000)
        >>> s(0.0)
        0.0
        >>> round(s(4000.0), 10)
        0.0006987712
        >>> round(s(1.0) * 1e7, 3)
        1.747
    """

    def __init__(
        self,
        d_model: float,
        warmup: float,
        scale: float = 1.0,
    ) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if warmup <= 0:
            raise ValueError(f"warmup must be positive, got {warmup}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.d_model = float(d_model)
        self.warmup = float(warmup)
        self.scale = float(scale)

    def __call__(self, epoch: float) -> float:
        """Return the learning rate at ``epoch``; ``0.0`` at ``epoch == 0``."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        if epoch == 0:
            return 0.0
        inner = min(epoch**-0.5, epoch * self.warmup**-1.5)
        return self.scale * self.d_model**-0.5 * inner

    def multiplier(self, epoch: float) -> float:
        """Return ``self(epoch) / self(0.0)``; 0 if ``self(0.0) == 0``."""
        base = self(0.0)
        if base == 0.0:
            return 0.0
        return self(epoch) / base
