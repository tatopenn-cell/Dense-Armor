"""Bayes-UCB policy, quantile of the posterior as the index.

The index of an arm is a high quantile of its posterior distribution of
the mean reward. The quantile level shrinks with the number of plays,
``1 - 1 / (t (log n)^c)``, where ``n`` is the horizon and ``c`` a
parameter. The paper proves a logarithmic regret bound for ``c >= 5``,
but recommends ``c = 0`` in practice.
"""

import math
from collections.abc import Sequence
from typing import Any

from scipy.stats import beta as _beta
from scipy.stats import norm as _norm
from scipy.stats import t as _student

from dense_armor.utility.bandit.base import BanditPolicy


class BayesUCB(BanditPolicy):
    """Bayes-UCB policy.

    At each step the policy computes, for each arm ``j``, the
    ``1 - 1 / (t (log n)^c)`` quantile of the posterior distribution of
    the mean reward, where ``t`` is the number of plays so far and ``n``
    the horizon. The arm with the highest quantile is played. For
    Bernoulli rewards the posterior is ``Beta(1 + S_j, 1 + F_j)``; a
    reward counts as a success when it is ``>= 0.5``, so a continuous
    reward in [0, 1] is binarised (the paper assumes rewards in {0, 1}). For
    Gaussian rewards with known variance the posterior is
    ``N(mu_j, sigma^2 / n_j)``; with ``estimate_var=True`` the posterior
    is a Student-t with ``n_j - 1`` degrees of freedom, as in Section 2
    of the paper.

    Args:
        arms: the sequence of arms.
        family: ``"bernoulli"`` or ``"gaussian"``.
        horizon: the horizon ``n`` used in the quantile level, integer
            or ``None`` (in which case ``n`` is set to ``n_seen_ + 1``).
        c: exponent in the quantile level, default ``0``.
        sigma: known standard deviation of Gaussian rewards.
        estimate_var: if True, use the Student-t posterior of the paper
            with unknown variance. Ignored for the Bernoulli family.
        seed: seed of the random generator.

    Raises:
        ValueError: if ``family`` is unknown or ``sigma <= 0`` for the
            Gaussian family.

    Examples:
        >>> from dense_armor.utility.bandit import BayesUCB
        >>> p = BayesUCB(arms=[0, 1], family="bernoulli", horizon=100, seed=0)
        >>> for _ in range(200):
        ...     a = p.pull()
        ...     _ = p.update(a, 1.0 if a == 1 else 0.0)
        >>> p.pull()
        1

    References:
        Kaufmann, E., Cappe, O., Garivier, A. (2012). On Bayesian upper
            confidence bounds for bandit problems. In AISTATS 15.
            Algorithm 1, page 3: the quantile schedule
            ``q_j(t) = Q(1 - 1/(t (log n)^c), lambda_j^{t-1})``. The
            Bernoulli posterior is on page 3, the Gaussian and
            Student-t posteriors on page 4.
    """

    def __init__(
        self,
        arms: Sequence[Any],
        family: str = "bernoulli",
        horizon: int | None = None,
        c: float = 0.0,
        sigma: float = 1.0,
        estimate_var: bool = False,
        seed: int = 0,
    ) -> None:
        """Initialise the policy.

        Args:
            arms: the sequence of arms.
            family: ``"bernoulli"`` or ``"gaussian"``.
            horizon: the horizon ``n``, or ``None``.
            c: exponent of the quantile level.
            sigma: known standard deviation of Gaussian rewards.
            estimate_var: whether to use the Student-t posterior.
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
        self.horizon = horizon
        self.c = float(c)
        self.sigma = float(sigma)
        self.estimate_var = bool(estimate_var)
        self.counts_: dict[Any, int] = {a: 0 for a in self.arms}
        self.means_: dict[Any, float] = {a: 0.0 for a in self.arms}
        self.successes_: dict[Any, int] = {a: 0 for a in self.arms}
        self.m2_: dict[Any, float] = {a: 0.0 for a in self.arms}

    def _quantile(self, arm: Any, t: int) -> float:
        if self.family == "bernoulli":
            s = self.successes_[arm] + 1.0
            f = self.counts_[arm] - self.successes_[arm] + 1.0
            n_h = self.horizon if self.horizon else (self.n_seen_ + 1)
            alpha = 1.0 / (t * (math.log(max(n_h, 2)) ** self.c))
            return float(_beta.ppf(1.0 - alpha, s, f))
        nj = self.counts_[arm]
        if nj == 0:
            n_h = self.horizon if self.horizon else (self.n_seen_ + 1)
            alpha = 1.0 / (t * (math.log(max(n_h, 2)) ** self.c))
            return float(_norm.ppf(1.0 - alpha, 0.0, 1.0))
        n_h = self.horizon if self.horizon else (self.n_seen_ + 1)
        alpha = 1.0 / (t * (math.log(max(n_h, 2)) ** self.c))
        if self.estimate_var and nj >= 2:
            var = max(self.m2_[arm], 0.0) / (nj - 1)
            sd = math.sqrt(var / nj)
            q = float(_student.ppf(1.0 - alpha, nj - 1))
            return self.means_[arm] + q * sd
        sd = self.sigma / math.sqrt(nj)
        q = float(_norm.ppf(1.0 - alpha, 0.0, 1.0))
        return self.means_[arm] + q * sd

    def pull(self, context: dict | None = None) -> Any:
        """Choose an arm.

        Args:
            context: unused.

        Returns:
            The arm with the highest posterior quantile.
        """
        t = max(self.n_seen_ + 1, 1)
        return max(self.arms, key=lambda a: self._quantile(a, t))

    def update(
        self,
        arm: Any,
        reward: float,
        context: dict | None = None,
    ) -> "BayesUCB":
        """Learn from the reward of ``arm``.

        Args:
            arm: the arm that was pulled.
            reward: the observed reward.
            context: unused.

        Returns:
            The policy itself.
        """
        if not self._valid(arm, reward):
            self.n_missing_ += 1
            return self
        r = float(reward)
        n = self.counts_[arm] + 1
        self.counts_[arm] = n
        m = self.means_[arm]
        self.means_[arm] = m + (r - m) / n
        if self.family == "bernoulli" and r >= 0.5:
            self.successes_[arm] += 1
        if self.family == "gaussian":
            self.m2_[arm] = self.m2_[arm] + (r - m) * (r - self.means_[arm])
        self.n_seen_ += 1
        return self
