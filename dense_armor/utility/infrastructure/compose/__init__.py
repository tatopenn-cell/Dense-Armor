"""Compose: small transformers to select, drop and reshape channels."""

from dense_armor.utility.infrastructure.compose.func import FuncTransformer
from dense_armor.utility.infrastructure.compose.select import Discard, Select

__all__ = [
    "Discard",
    "FuncTransformer",
    "Select",
]
