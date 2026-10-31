"""AdamW, Adam with decoupled weight decay."""

from typing import Any

import numpy as np

from dense_armor.utility.optim.adam import Adam


class AdamW(Adam):
    """Adam with weight decay decoupled from the gradient.

    One update, following Algorithm 2 of Loshchilov and Hutter (2019),
    page 3, without line 6 (line 6 adds ``lambda * theta`` to the gradient,
    which is the L2 variant of the same algorithm):

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - eta_t * (alpha * m_hat / (sqrt(v_hat) + eps)
                                         + lambda * theta_{t-1})

    ``eta_t`` is the schedule multiplier of the paper. Here ``lr`` is the
    step size of the Adam term, and ``Trainer`` sets it at every step to the
    value of the schedule, so ``lr`` already contains ``eta_t``. The decay
    term is multiplied by ``eta``, the attribute that holds the multiplier
    of the decay: it is ``1.0`` until the caller sets it. A schedule that
    decays ``lr`` does not decay the weight decay of this class unless the
    caller sets ``eta`` to the same multiplier at every step. The decay is
    ``eta * weight_decay * theta`` and not ``lr * weight_decay * theta``
    (line 12 of the algorithm has no ``alpha`` in front of ``lambda``).

    Args:
        lr: step size of the Adam term, strictly positive.
        beta1: decay of the first moment, in [0, 1).
        beta2: decay of the second moment, in [0, 1).
        eps: numerical guard, strictly positive.
        weight_decay: the decay coefficient ``lambda``, not negative.

    Raises:
        ValueError: on a value outside the ranges above.

    References:
        Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay
        Regularization. ICLR 2019, arXiv:1711.05101. Algorithm 2 and
        Proposition 2, page 3.

        The decay is applied to every array the optimizer is given. A
        ``Trainer`` with ``decay="decoupled"`` decays only the arrays with
        two or more dimensions; the two give the same trajectory only when
        every parameter of the model is such an array and the multiplier is
        the same (our statement, tested in ``test/optim/test_adamw.py``).

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.optim import AdamW
        >>> opt = AdamW(lr=0.1, weight_decay=0.5)
        >>> w = opt.step(np.array([1.0]), np.array([1.0]))
        >>> bool(np.allclose(w, [0.4], atol=1e-6))
        True
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr, beta1, beta2, eps)
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        self.weight_decay = float(weight_decay)
        self.eta = 1.0

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """One update: the Adam step, then the decoupled decay of ``weights``."""
        adam_step = super().step(weights, gradient)
        return adam_step - self.eta * self.weight_decay * weights

    def update(
        self, state: dict[str, Any], weights: Any, gradient: Any
    ) -> tuple[dict[str, Any], Any]:
        """Pure form of :meth:`step` on JAX arrays."""
        new_state, adam_step = super().update(state, weights, gradient)
        return new_state, adam_step - self.eta * self.weight_decay * weights
