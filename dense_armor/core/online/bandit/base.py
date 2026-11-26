"""Base class for bandit policies."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from dense_armor.roles import Root


class _PullPolicy(Root):
    """Base of the classes that learn by choosing and rewarding, not from a table.

    A class of this kind has no ``learn_one``: it learns one reward at a
    time, and its settings are checked before they change in place.
    """

    _learn_with = "pull() and update(arm, reward)"
    _fit_tail = ", or judge a policy on a log with offline_replay"

    def fit(
        self,
        X: Any = None,
        y: Any = None,
        t: Any = None,
        epochs: int = 1,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> "_PullPolicy":
        """Refuse to be fitted on a table, and say how to learn.

        Raises:
            TypeError: always, with the way to learn.
        """
        raise TypeError(
            f"{type(self).__name__} is not fitted on a table: learn with "
            f"{self._learn_with}{self._fit_tail}"
        )

    def mutate(self, new_attrs: dict[str, Any]) -> "_PullPolicy":
        """Change, in place, the settings that are read at every decision.

        The names a class lists in ``_mutable_attributes`` are changed
        after the same check as in the constructor; a value that does not
        pass raises ``ValueError`` and nothing changes. The other names
        are left alone, as in ``Root.mutate``. The arms, the seed and
        what has been learned are never changed here.

        Args:
            new_attrs: mapping from a setting to its new value.

        Returns:
            The object itself.

        Raises:
            ValueError: if a mutable setting gets a value that the
                constructor would refuse.
        """
        clean = {}
        for name, value in new_attrs.items():
            if name in self._mutable_attributes and hasattr(self, name):
                value = self._check_setting(name, value)
            clean[name] = value
        super().mutate(clean)
        return self

    def _check_setting(self, name: str, value: Any) -> Any:
        """Check the new value of a mutable setting and return the value to store."""
        return value


class BanditPolicy(_PullPolicy):
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
