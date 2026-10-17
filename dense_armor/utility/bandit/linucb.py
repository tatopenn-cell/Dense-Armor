"""LinUCB with disjoint linear models.

One Bayesian linear regression per arm. The upper confidence bound of an
arm is the mean prediction plus ``alpha`` times the standard deviation
of the model, both taken at the current context. The algorithm is
Algorithm 1 of Li et al. (2010).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Root
from dense_armor.utility.linear.bayes import BayesianLinearRegression


class LinUCBDisjoint(Root):
    """LinUCB with disjoint linear models.

    Each arm has its own Bayesian linear regression. At each step the
    policy computes, for every arm ``a``,

    ``p_a = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x)``

    where ``theta_a`` is the posterior mean of the regression and
    ``A_a = I + sum_t x_t x_t^T`` the precision matrix, and plays the
    arm with the highest ``p_a``. The ``alpha`` is the confidence
    parameter of the paper (its only input), not the prior precision of
    the regression; the prior precision of the model is fixed to ``1.0``
    so that ``A_a`` matches the paper's ``I_d + X^T X``.

    Args:
        arms: the sequence of arms.
        alpha: confidence parameter, in ``(0, 1]`` in practice, any
            positive value.
        seed: kept for symmetry with the other policies, unused (the
            algorithm is deterministic).
        feature_keys: names of the context features to use, or ``None``
            to take the keys of the first context seen.

    Raises:
        ValueError: if ``alpha <= 0`` or ``arms`` is empty.

    Examples:
        >>> import numpy as np
        >>> from dense_armor.utility.bandit import LinUCBDisjoint
        >>> keys = ("bright", "const")
        >>> p = LinUCBDisjoint(arms=["red", "blue"], alpha=0.1,
        ...                    feature_keys=keys)
        >>> rng = np.random.default_rng(0)
        >>> for _ in range(300):
        ...     b = float(rng.uniform(0.0, 1.0))
        ...     x = {"bright": b, "const": 1.0}
        ...     a = p.pull(context=x)
        ...     r = (1.0 - b) if a == "red" else b
        ...     _ = p.update(a, r, context=x)
        >>> p.pull(context={"bright": 0.05, "const": 1.0})
        'red'
        >>> p.pull(context={"bright": 0.95, "const": 1.0})
        'blue'

    References:
        Li, L., Chu, W., Langford, J., Schapire, R. E. (2010). A
            contextual-bandit approach to personalized news article
            recommendation. In WWW 2010. Algorithm 1, page 4: the
            index ``p_{t,a} = theta_a^T x_{t,a}
            + alpha sqrt(x_{t,a}^T A_a^{-1} x_{t,a})`` and the rank-one
            update of ``A_a`` and ``b_a``.
    """

    budget_s = 1e-3
    memory_class = "O(n_arms * d^2)"

    def __init__(
        self,
        arms: Sequence[Any],
        alpha: float = 0.3,
        seed: int = 0,
        feature_keys: Sequence[str] | None = None,
    ) -> None:
        """Initialise the policy.

        Args:
            arms: the sequence of arms.
            alpha: confidence parameter.
            seed: unused, kept for symmetry.
            feature_keys: names of the context features, or ``None``.

        Raises:
            ValueError: if ``alpha <= 0`` or ``arms`` is empty.
        """
        super().__init__()
        if not arms:
            raise ValueError("arms must be non-empty")
        if float(alpha) <= 0.0:
            raise ValueError("alpha must be positive")
        self.arms = tuple(arms)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.feature_keys = tuple(feature_keys) if feature_keys is not None else None
        self.models_: dict[Any, BayesianLinearRegression] = {}
        self.n_seen_ = 0
        self.n_missing = 0

    def _model(self, arm: Any) -> BayesianLinearRegression:
        if arm not in self.models_:
            self.models_[arm] = BayesianLinearRegression(
                alpha=1.0,
                beta=1.0,
                intercept=False,
                feature_keys=self.feature_keys,
            )
        return self.models_[arm]

    def pull(self, context: dict | None = None) -> Any:
        """Choose an arm given a context.

        Args:
            context: dict of features. Required.

        Returns:
            The arm with the highest upper confidence bound.

        Raises:
            ValueError: if ``context`` is ``None`` or empty.
        """
        if not context:
            raise ValueError("context is required for LinUCBDisjoint")
        best_arm = self.arms[0]
        best_score = -np.inf
        for a in self.arms:
            model = self._model(a)
            mean, std = model.predict_one(context, return_std=True)
            var_ucb = max(float(std) ** 2 - 1.0, 0.0)
            score = float(mean) + self.alpha * math.sqrt(var_ucb)
            if score > best_score:
                best_score = score
                best_arm = a
        return best_arm

    def update(
        self,
        arm: Any,
        reward: float,
        context: dict | None = None,
    ) -> "LinUCBDisjoint":
        """Learn from the reward of ``arm``.

        Args:
            arm: the arm that was pulled.
            reward: the observed reward.
            context: the same context passed to ``pull``.

        Returns:
            The policy itself.
        """
        if arm not in self.arms or not context or reward is None:
            self.n_missing += 1
            return self
        try:
            if not np.isfinite(float(reward)):
                self.n_missing += 1
                return self
        except (TypeError, ValueError):
            self.n_missing += 1
            return self
        self._model(arm).learn_one(context, float(reward))
        self.n_seen_ += 1
        return self

    def _unit_test_skips(self) -> set[str]:
        """Return the set of checks that do not apply to bandit policies.

        Returns:
            A set with the names of the checks that do not apply.
        """
        return set()
