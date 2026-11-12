"""A ``Root`` whose ``state_dict`` also holds the private attributes named in ``_state_attrs``."""

import copy
from typing import Any

from dense_armor.roles import Root


class StatefulRoot(Root):
    """``Root.state_dict`` keeps only attributes that end with an underscore.

    The audio primitives keep their memory (delay lines, filter taps, buffers) in private
    attributes, so a subclass lists them in ``_state_attrs`` and they are saved and restored too.
    """

    _state_attrs: tuple[str, ...] = ()

    def state_dict(self) -> dict[str, Any]:
        """The learned state, including the attributes named in ``_state_attrs``."""
        state = super().state_dict()
        for k in self._state_attrs:
            state[k] = copy.deepcopy(getattr(self, k))
        return state
