"""Active learning strategies for evolving data streams."""

from dense_armor.core.online.active.base import ActiveStrategy
from dense_armor.core.online.active.strategies import (
    FixedUncertainty,
    RandomStrategy,
    VariableRandomizedUncertainty,
    VariableUncertainty,
)

__all__ = [
    "ActiveStrategy",
    "FixedUncertainty",
    "RandomStrategy",
    "VariableRandomizedUncertainty",
    "VariableUncertainty",
]
