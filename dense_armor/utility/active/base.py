"""Base class for active learning strategies."""

from typing import Any

import numpy as np

from dense_armor.roles import Root


class ActiveStrategy(Root):
    """Base class for active learning strategies on a stream.

    An active strategy decides, for each arriving instance, whether to
    request the true label. The decision uses the model's prediction and
    a labelling budget ``B`` in ``(0, 1]``, the fraction of instances
    that the strategy is allowed to label.

    Args:
        budget: labelling budget ``B``, in ``(0, 1]``.
        seed: seed of the random generator.

    Raises:
        ValueError: if ``budget`` is outside ``(0, 1]``.
    """

    budget_s = 1e-3
    memory_class = "O(1)"

    def __init__(self, budget: float = 0.1, seed: int = 0) -> None:
        """Initialise the strategy.

        Args:
            budget: labelling budget.
            seed: seed of the random generator.

        Raises:
            ValueError: if ``budget`` is outside ``(0, 1]``.
        """
        super().__init__()
        if not 0.0 < float(budget) <= 1.0:
            raise ValueError("budget must be in (0, 1]")
        self.budget = float(budget)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.n_seen_ = 0
        self.n_labelled_ = 0
        self.n_missing = 0

    def should_label(self, x: dict, model: Any) -> bool:
        """Decide whether to request the true label of ``x``.

        Args:
            x: the context.
            model: a classifier with ``predict_proba_one``.

        Returns:
            True if the strategy requests the label.

        Raises:
            NotImplementedError: in the base class.
        """
        raise NotImplementedError
