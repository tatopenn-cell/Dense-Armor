"""Linear scaling of the learning rate for data-parallel training.

The rule is Goyal et al. (2017), Section 2.1, page 2: when the minibatch
size is multiplied by ``k``, multiply the learning rate by ``k``. The
warmup of Section 2.2, page 3, goes with it: the learning rate starts
from the small minibatch value and grows linearly to the scaled one
over a number of epochs, so the model does not diverge at the start.
"""

from __future__ import annotations


def linear_scaling(lr: float, batch: int, base_batch: int) -> float:
    """Return ``lr * batch / base_batch``.

    The rule of Goyal et al. (2017), Section 2.1, page 2: "When the
    minibatch size is multiplied by ``k``, multiply the learning rate
    by ``k``. All other hyper-parameters (weight decay, etc.) are kept
    unchanged."

    Args:
        lr: the learning rate of the small ``base_batch``.
        batch: the larger minibatch size, at least ``base_batch``.
        base_batch: the reference minibatch size, positive.

    Returns:
        The scaled learning rate.

    Raises:
        ValueError: if ``base_batch`` is not positive, if ``batch`` is
            below ``base_batch``, or if ``lr`` is not positive.

    Examples:
        >>> from dense_armor.utility.neural.data_parallel import linear_scaling
        >>> linear_scaling(0.1, 256, 32)
        0.8
    """
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if base_batch <= 0:
        raise ValueError(f"base_batch must be positive, got {base_batch}")
    if batch < base_batch:
        raise ValueError(
            f"batch {batch} is below base_batch {base_batch}: the rule "
            "scales up, not down"
        )
    return float(lr) * int(batch) / int(base_batch)
