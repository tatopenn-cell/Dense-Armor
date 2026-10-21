"""Neural bricks: parameters, initialisations, activations, losses."""

from dense_armor.utility.neural.adapter import AsTransformer
from dense_armor.utility.neural.layers import (
    Dense,
    Dropout,
    Embedding,
    Residual,
    dense_backward,
    dense_forward,
)
from dense_armor.utility.neural.norm import BatchNorm, LayerNorm
from dense_armor.utility.neural.params import Param

__all__ = [
    "Param",
    "AsTransformer",
    "BatchNorm",
    "Dense",
    "Dropout",
    "Embedding",
    "LayerNorm",
    "Residual",
    "dense_backward",
    "dense_forward",
]

from dense_armor.utility.neural.chain import Parallel, Sequential
from dense_armor.utility.neural.models import mlp, resnet_mlp

__all__ = list(__all__) + ["Parallel", "Sequential", "mlp", "resnet_mlp"]
