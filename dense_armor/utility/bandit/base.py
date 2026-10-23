"""Base class for bandit policies."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Root


class BanditPolicy(Root):
    """Base class for bandit policies.

    A bandit policy chooses an arm at each step (``pull``) and learns from
    the reward of the chosen arm (``update``). Arms are any hashables. The
    set of arms is fixed at construction time; a ``pull`` returns one of
    those arms, and an ``update`` with an arm that is not in the set is
    counted in ``n_missing`` and ignored.

    Args:
        arms: the sequence of arms, any hashables.
        seed: seed of the random generator.

    Raises:
        ValueError: if ``arms`` is empty.

    Attributes:
        arms: the arms as a tuple.
        n_seen_: number of rewards learned so far.
        n_missing_: number of samples skipped because of NaN, missing arm,
            or non-numeric reward (``n_missing`` reads it).
    """

    budget_s = 1e-3
    memory_class = "O(n_arms)"

    def __init__(self, arms: Sequence[Any], seed: int = 0) -> None:
        """Initialise the policy.

        Args:
            arms: the sequence of arms.
            seed: seed of the random generator.

        Raises:
            ValueError: if ``arms`` is empty.
        """
        super().__init__()
        if not arms:
            raise ValueError("arms must be non-empty")
        self.arms = tuple(arms)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.n_seen_ = 0
        self.n_missing_ = 0

    @property
    def n_missing(self) -> int:
        return self.n_missing_

    def pull(self, context: dict | None = None) -> Any:
        """Choose an arm.

        Args:
            context: optional dict of features, ignored by non-contextual
                policies.

        Returns:
            The chosen arm.

        Raises:
            NotImplementedError: in the base class.
        """
        raise NotImplementedError

    def update(
        self, arm: Any, reward: float, context: dict | None = None
    ) -> "BanditPolicy":
        """Learn from the reward of ``arm``.

        Args:
            arm: the arm that was pulled.
            reward: the observed reward, a finite float.
            context: optional dict of features, ignored by non-contextual
                policies.

        Returns:
            The policy itself.

        Raises:
            NotImplementedError: in the base class.
        """
        raise NotImplementedError

    def _valid(self, arm: Any, reward: float | None = None) -> bool:
        """Return True when the arm is known and the reward is finite.

        Args:
            arm: the arm to check.
            reward: optional reward to check.

        Returns:
            True when the arm is known and every value is finite.
        """
        if arm not in self.arms:
            return False
        if reward is not None:
            try:
                if not np.isfinite(float(reward)):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _unit_test_skips(self) -> set[str]:
        """Return the set of checks that do not apply to bandit policies.

        Returns:
            A set with the names of the checks that do not apply.
        """
        return set()
