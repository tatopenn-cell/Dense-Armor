"""The four active learning strategies of Zliobaite et al. (2011)."""

import math
from typing import Any

from dense_armor.utility.active.base import ActiveStrategy


class RandomStrategy(ActiveStrategy):
    """Random strategy: label with probability ``B``.

    Baseline: with probability ``B`` the true label of the instance is
    requested, independently of the model. Algorithm 2 of the paper.

    References:
        Zliobaite, I., Bifet, A., Holmes, G., Pfahringer, B. (2011).
            Labeling strategies for active learning in data streams.
            Machine Learning 100(1). Algorithm 2, page 3: the budget
            is the probability that a sample is labelled.
    """

    def should_label(self, x: dict, model: Any) -> bool:
        """Decide whether to request the label.

        Args:
            x: the context, unused.
            model: the classifier, unused.

        Returns:
            True with probability ``budget``.
        """
        self.n_seen_ += 1
        ask = bool(self._rng.random() < self.budget)
        if ask:
            self.n_labelled_ += 1
        return ask


class FixedUncertainty(ActiveStrategy):
    """Fixed-threshold uncertainty strategy.

    Label the instance when the model's confidence in its best guess is
    below a fixed threshold ``theta``: ``P(y_hat | x) < theta``. No
    budget is enforced (the paper notes the fixed strategy can exhaust
    the budget or starve). Algorithm 3 of the paper.

    Args:
        theta: confidence threshold, in ``(0, 1)``.
        budget: kept for interface, not enforced.
        seed: seed of the random generator.

    References:
        Zliobaite et al. (2011). Algorithm 3, page 4.
    """

    def __init__(self, theta: float = 0.5, budget: float = 0.1, seed: int = 0) -> None:
        """Initialise the strategy.

        Args:
            theta: confidence threshold.
            budget: labelling budget, kept for interface.
            seed: seed of the random generator.
        """
        super().__init__(budget=budget, seed=seed)
        self.theta = float(theta)

    def should_label(self, x: dict, model: Any) -> bool:
        """Decide whether to request the label.

        Args:
            x: the context.
            model: the classifier.

        Returns:
            True if the confidence in the best guess is below ``theta``.
        """
        self.n_seen_ += 1
        p = model.predict_proba_one(x)
        if not p:
            self.n_labelled_ += 1
            return True
        conf = max(p.values())
        ask = float(conf) < self.theta
        if ask:
            self.n_labelled_ += 1
        return ask


class VariableUncertainty(ActiveStrategy):
    """Variable-threshold uncertainty strategy.

    The threshold starts at 1 and adjusts on every sample so the fraction
    of labelled samples stays close to the budget: after a labelled
    sample the threshold shrinks by ``(1 - s)``, after an unlabelled one
    it grows by ``(1 + s)``. Budget enforced: ``u / t < B``. Algorithm 4
    of the paper.

    Args:
        budget: labelling budget.
        s: adjusting step, small positive float.
        seed: seed of the random generator.

    References:
        Zliobaite et al. (2011). Algorithm 4, page 5.
    """

    def __init__(self, budget: float = 0.1, s: float = 0.01, seed: int = 0) -> None:
        """Initialise the strategy.

        Args:
            budget: labelling budget.
            s: adjusting step.
            seed: seed of the random generator.
        """
        super().__init__(budget=budget, seed=seed)
        self.s = float(s)
        self.theta_ = 1.0

    def should_label(self, x: dict, model: Any) -> bool:
        """Decide whether to request the label.

        Args:
            x: the context.
            model: the classifier.

        Returns:
            True if the instance is uncertain and the budget allows it.
        """
        self.n_seen_ += 1
        if (self.n_labelled_ / self.n_seen_) < self.budget:
            p = model.predict_proba_one(x)
            conf = max(p.values()) if p else 0.0
            if float(conf) < self.theta_:
                self.n_labelled_ += 1
                self.theta_ *= 1.0 - self.s
                return True
            self.theta_ *= 1.0 + self.s
        return False


class VariableRandomizedUncertainty(VariableUncertainty):
    """Variable-threshold uncertainty with a randomised threshold.

    Same threshold schedule as :class:`VariableUncertainty`, but before
    comparing the confidence the threshold is multiplied by
    ``eta ~ Normal(1, delta)``. The randomisation occasionally labels
    instances far from the decision boundary, which the paper argues
    catches concept drift in regions the plain uncertainty strategy
    ignores. Algorithm 5 of the paper.

    Args:
        budget: labelling budget.
        s: adjusting step.
        delta: variance of the random multiplier, positive, as the paper
            states it. The default 0.01 is my choice. With a large
            ``delta`` the multiplier ``eta`` is negative with probability
            ``P(Normal(1, delta) < 0)`` (about 16 % for ``delta = 1``);
            then the threshold ``theta * eta`` is negative and no label is
            requested at that step.
        seed: seed of the random generator.

    References:
        Zliobaite et al. (2011). Algorithm 5, page 6.
    """

    def __init__(
        self,
        budget: float = 0.1,
        s: float = 0.01,
        delta: float = 0.01,
        seed: int = 0,
    ) -> None:
        """Initialise the strategy.

        Args:
            budget: labelling budget.
            s: adjusting step.
            delta: variance of ``eta``.
            seed: seed of the random generator.

        Raises:
            ValueError: if ``delta <= 0``.
        """
        super().__init__(budget=budget, s=s, seed=seed)
        if float(delta) <= 0.0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)

    def should_label(self, x: dict, model: Any) -> bool:
        """Decide whether to request the label.

        Args:
            x: the context.
            model: the classifier.

        Returns:
            True if the instance is uncertain under the randomised
            threshold and the budget allows it.
        """
        self.n_seen_ += 1
        if (self.n_labelled_ / self.n_seen_) < self.budget:
            p = model.predict_proba_one(x)
            conf = max(p.values()) if p else 0.0
            eta = float(self._rng.normal(1.0, math.sqrt(self.delta)))
            if float(conf) < self.theta_ * eta:
                self.n_labelled_ += 1
                self.theta_ *= 1.0 - self.s
                return True
            self.theta_ *= 1.0 + self.s
        return False
