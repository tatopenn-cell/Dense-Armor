"""Shared base of the bandit policies and of the active learning strategies."""

from typing import Any

from dense_armor.roles import Root


class _PullPolicy(Root):
    """Base of the classes that decide one item at a time, not from a table.

    A class of this kind has no ``learn_one``: it learns one reward at a
    time, and its settings are checked before they change in place.
    """

    _fit_hint = (
        "learn with pull() and update(arm, reward), or judge a policy on a log "
        "with offline_replay"
    )

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
            f"{type(self).__name__} is not fitted on a table: {self._fit_hint}"
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

    def set_params(self, **params: Any) -> "_PullPolicy":
        """Set hyper-parameters in place, checking the mutable ones.

        The names a class lists in ``_mutable_attributes`` are checked as in
        the constructor before anything is set, so a value that does not pass
        raises ``ValueError`` and nothing changes. The other names are set as
        in ``Root.set_params``.

        Args:
            **params: the parameters to set.

        Returns:
            The object itself.

        Raises:
            ValueError: if a mutable setting gets a value that the
                constructor would refuse, or a name is not a parameter.
        """
        checked = {
            name: self._check_setting(name, value) if name in self._mutable_attributes else value
            for name, value in params.items()
        }
        super().set_params(**checked)
        return self

    def _check_setting(self, name: str, value: Any) -> Any:
        """Check the new value of a mutable setting and return the value to store."""
        return value
