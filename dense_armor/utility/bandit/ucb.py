"""Epsilon-greedy, UCB1 and UCB1-Tuned bandit policies.

The three policies come from Auer, Cesa-Bianchi and Fischer (2002).
The formulas follow the paper as cited.
"""

import math
from collections.abc import Sequence
from typing import Any

from dense_armor.utility.bandit.base import BanditPolicy


class EpsilonGreedy(BanditPolicy):
    """Epsilon-greedy policy.

    With probability ``1 - epsilon`` the policy plays the arm with the
    highest running mean reward; otherwise it plays an arm chosen
    uniformly at random. Each arm is played once before the first choice,
    so that every arm has a sample even with ``epsilon = 0``. The paper
    shows that a constant ``epsilon`` gives a linear regret; the
    logarithmic bound requires ``epsilon_n = c K / (d^2 n)``, where
    ``d`` is a lower bound on the gap to the best arm. This class keeps
    ``epsilon`` constant, which is the baseline the paper compares
    against.

    Args:
        arms: the sequence of arms.
        epsilon: exploration probability, in ``[0, 1]``.
        seed: seed of the random generator.

    Raises:
        ValueError: if ``epsilon`` is outside ``[0, 1]``.

    Examples:
        >>> from dense_armor.utility.bandit import EpsilonGreedy
        >>> p = EpsilonGreedy(arms=[0, 1], epsilon=0.0, seed=0)
        >>> for _ in range(20):
        ...     a = p.pull()
        ...     _ = p.update(a, 1.0 if a == 1 else 0.0)
        >>> p.pull()
        1

    References:
        Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). Finite-time
            analysis of the multiarmed bandit problem. Machine Learning
            47(2-3), 235-256. Section 3, "epsilon-greedy rule", page 5;
            Figure 3 and Theorem 3, pages 5-6; the choice
            ``epsilon_n = cK / (d^2 n)`` on page 10.
    """

    def __init__(
        self,
        arms: Sequence[Any],
        epsilon: float = 0.1,
        seed: int = 0,
    ) -> None:
        """Initialise the policy.

        Args:
            arms: the sequence of arms.
            epsilon: exploration probability, in ``[0, 1]``.
            seed: seed of the random generator.

        Raises:
            ValueError: if ``epsilon`` is outside ``[0, 1]``.
        """
        super().__init__(arms=arms, seed=seed)
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = float(epsilon)
        self.counts_: dict[Any, int] = {a: 0 for a in self.arms}
        self.means_: dict[Any, float] = {a: 0.0 for a in self.arms}

    def pull(self, context: dict | None = None) -> Any:
        """Choose an arm.

        Plays every arm once (round-robin over the untried arms), then
        explores with probability ``epsilon`` and otherwise plays the
        arm with the highest running mean.

        Args:
            context: unused.

        Returns:
            The chosen arm.
        """
        for a in self.arms:
            if self.counts_[a] == 0:
                return a
        if self._rng.random() < self.epsilon:
            return self.arms[int(self._rng.integers(len(self.arms)))]
        best = max(self.means_.values())
        for a in self.arms:
            if self.means_[a] == best:
                return a
        return self.arms[0]

    def update(
        self,
        arm: Any,
        reward: float,
        context: dict | None = None,
    ) -> "EpsilonGreedy":
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
        n = self.counts_[arm] + 1
        self.counts_[arm] = n
        m = self.means_[arm]
        self.means_[arm] = m + (float(reward) - m) / n
        self.n_seen_ += 1
        return self


class UCB1(BanditPolicy):
    """UCB1 policy.

    Each arm is played once; after that the policy plays the arm with the
    highest index ``mean_j + sqrt(2 * ln(n) / n_j)``, where ``n`` is the
    total number of plays and ``n_j`` is the number of times arm ``j``
    was played. When the rewards are in ``[0, 1]`` and ``mu*`` is the
    best expected reward, Theorem 1 of the paper bounds the expected
    regret after ``n`` plays by

    ``8 * sum_{i: mu_i < mu*} ln(n) / Delta_i
        + (1 + pi^2 / 3) * sum_j Delta_j``

    with ``Delta_i = mu* - mu_i``.

    Args:
        arms: the sequence of arms.
        seed: seed of the random generator.

    Examples:
        >>> from dense_armor.utility.bandit import UCB1
        >>> p = UCB1(arms=[0, 1], seed=0)
        >>> for _ in range(60):
        ...     a = p.pull()
        ...     _ = p.update(a, 1.0 if a == 1 else 0.0)
        >>> p.pull()
        1

    References:
        Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). Finite-time
            analysis of the multiarmed bandit problem. Machine Learning
            47(2-3), 235-256. Figure 1 and Theorem 1, page 3; the index
            ``sqrt(2 ln(n) / n_j)`` is stated explicitly on page 11.
    """

    def __init__(self, arms: Sequence[Any], seed: int = 0) -> None:
        """Initialise the policy.

        Args:
            arms: the sequence of arms.
            seed: seed of the random generator.
        """
        super().__init__(arms=arms, seed=seed)
        self.counts_: dict[Any, int] = {a: 0 for a in self.arms}
        self.means_: dict[Any, float] = {a: 0.0 for a in self.arms}

    def _index(self, arm: Any, n: int) -> float:
        """Return the UCB1 index of ``arm`` after ``n`` plays.

        Args:
            arm: the arm.
            n: total number of plays.

        Returns:
            ``mean_j + sqrt(2 ln(n) / n_j)``, or ``+inf`` if ``n_j == 0``.
        """
        nj = self.counts_[arm]
        if nj == 0:
            return float("inf")
        return self.means_[arm] + math.sqrt(2.0 * math.log(n) / nj)

    def pull(self, context: dict | None = None) -> Any:
        """Choose an arm.

        Plays every arm once (round-robin over the untried arms), then
        chooses the arm with the highest index.

        Args:
            context: unused.

        Returns:
            The chosen arm.
        """
        for a in self.arms:
            if self.counts_[a] == 0:
                return a
        n = self.n_seen_
        return max(self.arms, key=lambda a: self._index(a, n))

    def update(
        self,
        arm: Any,
        reward: float,
        context: dict | None = None,
    ) -> "UCB1":
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
        n = self.counts_[arm] + 1
        self.counts_[arm] = n
        m = self.means_[arm]
        self.means_[arm] = m + (float(reward) - m) / n
        self.n_seen_ += 1
        return self


class UCB1Tuned(UCB1):
    """UCB1-Tuned policy.

    Same as :class:`UCB1`, but the exploration bonus is
    ``sqrt((ln(n) / n_j) * min(1/4, V_j(n_j)))`` with
    ``V_j(s) = (1/s) * sum_{tau=1}^s X_{j,tau}^2 - mean_j(s)^2
              + sqrt(2 ln(n) / s)``,
    the upper confidence bound on the variance of arm ``j``. The paper
    reports better empirical performance than UCB1 but no proven bound.

    Args:
        arms: the sequence of arms.
        seed: seed of the random generator.

    Examples:
        >>> from dense_armor.utility.bandit import UCB1Tuned
        >>> p = UCB1Tuned(arms=[0, 1], seed=0)
        >>> for _ in range(60):
        ...     a = p.pull()
        ...     _ = p.update(a, 1.0 if a == 1 else 0.0)
        >>> p.pull()
        1

    References:
        Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). Finite-time
            analysis of the multiarmed bandit problem. Machine Learning
            47(2-3), 235-256. Section 4, "Experiments", page 11: the
            definition of ``V_j(s)`` and the substitution of
            ``sqrt(2 ln(n) / n_j)`` with
            ``sqrt((ln n / n_j) * min{1/4, V_j(n_j)})``.
    """

    def __init__(self, arms: Sequence[Any], seed: int = 0) -> None:
        """Initialise the policy.

        Args:
            arms: the sequence of arms.
            seed: seed of the random generator.
        """
        super().__init__(arms=arms, seed=seed)
        self.sumsq_: dict[Any, float] = {a: 0.0 for a in self.arms}

    def _v(self, arm: Any, n: int) -> float:
        """Return ``V_j(n_j)`` for arm ``arm`` after ``n`` total plays.

        Args:
            arm: the arm.
            n: total number of plays.

        Returns:
            The upper confidence bound on the variance of ``arm``.
        """
        nj = self.counts_[arm]
        if nj < 2:
            return 0.25
        mean = self.means_[arm]
        var = self.sumsq_[arm] / nj - mean * mean
        var = max(var, 0.0)
        bound = math.sqrt(2.0 * math.log(max(n, 2)) / nj)
        return var + bound

    def _index(self, arm: Any, n: int) -> float:
        """Return the UCB1-Tuned index of ``arm`` after ``n`` plays.

        Args:
            arm: the arm.
            n: total number of plays.

        Returns:
            ``mean_j + sqrt((ln n / n_j) * min(1/4, V_j(n_j)))``, or
            ``+inf`` if ``n_j == 0``.
        """
        nj = self.counts_[arm]
        if nj == 0:
            return float("inf")
        v = min(0.25, self._v(arm, n))
        return self.means_[arm] + math.sqrt(math.log(max(n, 2)) / nj * v)

    def update(
        self,
        arm: Any,
        reward: float,
        context: dict | None = None,
    ) -> "UCB1Tuned":
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
        self.sumsq_[arm] = self.sumsq_[arm] + r * r
        return super().update(arm, reward, context)
