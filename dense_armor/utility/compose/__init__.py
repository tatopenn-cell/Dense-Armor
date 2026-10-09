"""Compose: small transformers to select, drop and reshape channels."""

from dense_armor.utility.compose.func import FuncTransformer
from dense_armor.utility.compose.select import Discard, Select

__all__ = [
    "Discard",
    "FuncTransformer",
    "Select",
]
