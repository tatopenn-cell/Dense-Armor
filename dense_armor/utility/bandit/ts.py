"""Thompson sampling, Beta-Bernoulli and Gaussian families.

Thompson sampling draws a sample from the posterior of each arm's mean
and plays the arm with the highest draw. The two families implemented
here follow Chapelle and Li (2011).
"""

import math
from collections.abc import Sequence
from typing import Any

from dense_armor.utility.bandit.base import BanditPolicy


class ThompsonSampling(BanditPolicy):
    """Thompson sampling for a Bernoulli or a Gaussian bandit.

    At each step the policy draws a sample from the posterior of each
    arm's mean reward and plays the arm with the highest draw. For
    Bernoulli rewards the posterior is ``Beta(1 + S_j, 1 + F_j)`` with
    ``S_j`` and ``F_j`` the number of successes and failures of arm
    ``j`` (Beta(1, 1) prior, as the paper uses in its simulations); a
    reward counts as a success when it is ``>= 0.5``, so a continuous
    reward in [0, 1] is binarised. For Gaussian rewards with known
    variance ``sigma^2`` the posterior is ``N(mu_j, sigma^2 / n_j)``
    (a flat prior: ``mu_j`` is the running mean, not the mean under a
    ``N(0, 1)`` prior, and ``n_j`` the number of pulls); an arm never
    pulled is drawn from ``N(0, 1)``.

    Args:
        arms: the sequence of arms.
        family: ``"bernoulli"`` or ``"gaussian"``.
        sigma: known standard deviation of the Gaussian rewards. Ignored
            for the Bernoulli family.
        seed: seed of the random generator.

    Raises:
        ValueError: if ``family`` is unknown or ``sigma <= 0`` for the
            Gaussian family.

    Examples:
        >>> from dense_armor.utility.bandit import ThompsonSampling
        >>> p = ThompsonSampling(arms=[0, 1], family="bernoulli", seed=0)
        >>> for _ in range(200):
        ...     a = p.pull()
        ...     _ = p.update(a, 1.0 if a == 1 else 0.0)
        >>> p.pull()
        1

    References:
        Chapelle, O., Li, L. (2011). An empirical evaluation of Thompson
            sampling. In NIPS 24. Algorithm 2, page 2: Thompson sampling
            for the Bernoulli bandit with a Beta(a, b) prior; the
            Gaussian posterior is on pages 5-6, algorithm 3.
    """

    def __init__(
        self,
        arms: Sequence[Any],
        family: str = "bernoulli",
        sigma: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Initialise the policy.

        Args:
            arms: the sequence of arms.
            family: ``"bernoulli"`` or ``"gaussian"``.
            sigma: known standard deviation of Gaussian rewards.
            seed: seed of the random generator.

        Raises:
            ValueError: if ``family`` is unknown or ``sigma <= 0`` for
                the Gaussian family.
        """
        super().__init__(arms=arms, seed=seed)
        if family not in ("bernoulli", "gaussian"):
            raise ValueError("family must be 'bernoulli' or 'gaussian'")
        if family == "gaussian" and float(sigma) <= 0.0:
            raise ValueError("sigma must be positive for gaussian")
        self.family = family
        self.sigma = float(sigma)
        self.counts_: dict[Any, int] = {a: 0 for a in self.arms}
        self.means_: dict[Any, float] = {a: 0.0 for a in self.arms}
        self.successes_: dict[Any, int] = {a: 0 for a in self.arms}

    def _draw(self, arm: Any) -> float:
        if self.family == "bernoulli":
            s = self.successes_[arm] + 1.0
            f = self.counts_[arm] - self.successes_[arm] + 1.0
            return float(self._rng.beta(s, f))
        nj = self.counts_[arm]
        if nj == 0:
            return float(self._rng.normal(0.0, 1.0))
        return float(self._rng.normal(self.means_[arm], self.sigma / math.sqrt(nj)))

    def pull(self, context: dict | None = None) -> Any:
        """Choose an arm.

        Args:
            context: unused.

        Returns:
            The arm with the highest posterior sample.
        """
        return max(self.arms, key=self._draw)

    def update(
        self,
        arm: Any,
        reward: float,
        context: dict | None = None,
    ) -> "ThompsonSampling":
        """Learn from the reward of ``arm``.

        Args:
            arm: the arm that was pulled.
            reward: the observed reward.
            context: unused.

        Returns:
            The policy itself.
        """
        if not self._valid(arm, reward):
            self.n_missing += 1
            return self
        r = float(reward)
        n = self.counts_[arm] + 1
        self.counts_[arm] = n
        m = self.means_[arm]
        self.means_[arm] = m + (r - m) / n
        if self.family == "bernoulli" and r >= 0.5:
            self.successes_[arm] += 1
        self.n_seen_ += 1
        return self
