"""Exp3 policy for the non-stochastic multi-armed bandit.

Exp3 mixes the uniform distribution with an exponential weight on the
estimated cumulative reward of each arm. The estimate divides the
observed reward by the probability that the arm was played, so it is
unbiased for any assignment of rewards, adversarial included.
"""

import math
from collections.abc import Sequence
from typing import Any

from dense_armor.utility.bandit.base import BanditPolicy


class Exp3(BanditPolicy):
    """Exp3 policy for the non-stochastic multi-armed bandit.

    At each step the policy computes, for each arm ``i``,

    ``p_i(t) = (1 - gamma) * w_i(t) / sum_j w_j(t) + gamma / K``

    draws the arm according to ``p(t)``, receives a reward
    ``x_i(t)`` in ``[0, 1]``, builds the unbiased estimate
    ``x_hat_i(t) = x_i(t) / p_i(t)`` for the drawn arm and ``0`` for
    the others, and updates the weight

    ``w_i(t+1) = w_i(t) * exp(gamma * x_hat_i(t) / K)``.

    The weight vector is rescaled by its maximum after every update, so
    the numbers stay finite; the distribution ``p`` is not affected by a
    uniform rescaling.

    Args:
        arms: the sequence of arms.
        gamma: mixing coefficient, in ``(0, 1]``. ``None`` (default)
            uses the value that Corollary 3.2 recommends for a known
            horizon ``T`` and rewards in ``[0, 1]``, that is
            ``min(1, sqrt(K ln K / ((e-1) T)))``, which needs ``T``.
            When ``horizon`` is ``None`` and ``gamma`` is ``None``, the
            constructor raises.
        horizon: the horizon ``T`` used to compute the default
            ``gamma``. Ignored when ``gamma`` is given.
        seed: seed of the random generator.

    Raises:
        ValueError: if ``gamma`` is outside ``(0, 1]``, or if both
            ``gamma`` and ``horizon`` are ``None``.

    Examples:
        >>> from dense_armor.utility.bandit import Exp3
        >>> p = Exp3(arms=[0, 1], gamma=0.1, seed=0)
        >>> for _ in range(400):
        ...     a = p.pull()
        ...     _ = p.update(a, 1.0 if a == 1 else 0.0)
        >>> p.pull()
        1

    References:
        Auer, P., Cesa-Bianchi, N., Freund, Y., Schapire, R. E. (2002).
            The nonstochastic multiarmed bandit problem. SIAM Journal
            on Computing 32(1), 48-77. Figure 1, page 6 (the
            pseudo-code); Theorem 3.1 and Corollary 3.2, page 7 (the
            regret bound and the choice of ``gamma``).
    """

    def __init__(
        self,
        arms: Sequence[Any],
        gamma: float | None = None,
        horizon: int | None = None,
        seed: int = 0,
    ) -> None:
        """Initialise the policy.

        Args:
            arms: the sequence of arms.
            gamma: mixing coefficient, or ``None`` to use the default.
            horizon: horizon ``T`` for the default ``gamma``.
            seed: seed of the random generator.

        Raises:
            ValueError: if ``gamma`` is outside ``(0, 1]``, or if both
                ``gamma`` and ``horizon`` are ``None``.
        """
        super().__init__(arms=arms, seed=seed)
        if gamma is None:
            if horizon is None:
                raise ValueError("gamma or horizon must be given to fix the mixing")
            gamma = min(
                1.0,
                math.sqrt(
                    len(self.arms)
                    * math.log(len(self.arms))
                    / ((math.e - 1.0) * float(horizon))
                ),
            )
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = float(gamma)
        self.horizon = horizon
        self.weights_: dict[Any, float] = {a: 1.0 for a in self.arms}

    def _probabilities(self) -> dict[Any, float]:
        total = sum(self.weights_.values())
        k = len(self.arms)
        if total <= 0.0:
            return {a: 1.0 / k for a in self.arms}
        return {
            a: (1.0 - self.gamma) * self.weights_[a] / total + self.gamma / k
            for a in self.arms
        }

    def pull(self, context: dict | None = None) -> Any:
        """Choose an arm.

        Args:
            context: unused.

        Returns:
            The chosen arm.
        """
        p = self._probabilities()
        r = float(self._rng.random())
        cum = 0.0
        for a in self.arms:
            cum += p[a]
            if r <= cum:
                return a
        return self.arms[-1]

    def update(
        self,
        arm: Any,
        reward: float,
        context: dict | None = None,
    ) -> "Exp3":
        """Learn from the reward of ``arm``.

        Args:
            arm: the arm that was pulled.
            reward: the observed reward, in ``[0, 1]``.
            context: unused.

        Returns:
            The policy itself.
        """
        if not self._valid(arm, reward):
            self.n_missing += 1
            return self
        x = float(reward)
        p_arm = self._probabilities()[arm]
        x_hat = x / p_arm if p_arm > 0.0 else 0.0
        self.weights_[arm] *= math.exp(self.gamma * x_hat / len(self.arms))
        w_max = max(self.weights_.values())
        if w_max > 1e100:
            for a in self.arms:
                self.weights_[a] /= w_max
        self.n_seen_ += 1
        return self
