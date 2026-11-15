"""VSAEstimator: a streaming VSA solver, compatible with Dense-Armor's
prequential / OnlineModelSelection protocol.

The estimator accumulates training pairs and re-learns a VSAProgram on
every ``learn_one``. ``predict_one`` returns the deduced output grid.
Each instance carries its own object hypothesis (a ``VSAVariant``),
so ``OnlineModelSelection`` can compare the property hypotheses on the
same stream.

This is the property-selection step of Joffe & Eliasmith (2025),
Section 3.2.2, expressed with Dense-Armor's prequential cross-validation
instead of the paper's leave-one-out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.vsa_objects import ObjectEncoder
from dense_armor.core.graph.vsa_solver import VSAProgram
from dense_armor.core.graph.vsa_variant import (
    VSAVariant,
    learn_variant,
    predict_variant,
)

_log = logging.getLogger(__name__)


@dataclass
class VSAEstimator:
    """A VSA solver that learns one pair at a time.

    Args:
        encoder: the shared VSA encoder.
        variant_name: which property hypothesis to use.
    """

    encoder: ObjectEncoder
    variant_name: str = "all"
    pairs: list[dict] = field(default_factory=list)
    program: VSAProgram | None = None

    def _variant(self) -> VSAVariant:
        return VSAVariant(self.variant_name, self.encoder)

    def learn_one(self, x, y=None, t=None):
        """Add a training pair and re-learn the program.

        Args:
            x: the input pair ``{"input": ..., "output": ...}``.
            y: unused (kept for the protocol).
            t: unused (kept for the protocol).

        Returns:
            self.
        """
        self.pairs.append(x)
        task = {"train": self.pairs, "test": []}
        try:
            self.program = learn_variant(task, self._variant())
        except Exception as exc:
            _log.warning("VSA learning failed, no program: %r", exc)
            self.program = None
        return self

    def predict_one(self, x, t=None):
        """Predict the output for a test input.

        Args:
            x: the test sample. Either a dict with an ``"input"`` key
                (the ARC pair protocol) or a raw grid (list of lists or
                ``Grid``).
            t: unused.

        Returns:
            the predicted grid, or a zeros grid if nothing was learned.
        """
        if isinstance(x, dict) and "input" in x:
            raw = x["input"]
        else:
            raw = x
        g = raw if isinstance(raw, Grid) else Grid.from_list(raw)
        if self.program is None:
            return Grid.from_list([[0] * g.width for _ in range(g.height)])
        return predict_variant(self.program, g, self._variant())
