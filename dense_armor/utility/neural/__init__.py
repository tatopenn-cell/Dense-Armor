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

from dense_armor.utility.neural.attention import (
    MultiHeadSelfAttention,
    TransformerBlock,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    sinusoidal_positions,
)
from dense_armor.utility.neural.chain import Parallel, Sequential
from dense_armor.utility.neural.conv import (
    AvgPool1D,
    AvgPool2D,
    Conv1D,
    Conv2D,
    Flatten,
    MaxPool1D,
    MaxPool2D,
)
from dense_armor.utility.neural.models import (
    AddPositions,
    LastStep,
    MeanOverTime,
    cnn,
    mlp,
    recurrent_net,
    resnet_mlp,
    transformer_encoder,
)
from dense_armor.utility.neural.recurrent import GRU, LSTM, RNN

__all__ = list(__all__) + [
    "Parallel",
    "Sequential",
    "mlp",
    "resnet_mlp",
    "AddPositions",
    "LastStep",
    "MeanOverTime",
    "cnn",
    "recurrent_net",
    "transformer_encoder",
    "Conv1D",
    "Conv2D",
    "MaxPool1D",
    "MaxPool2D",
    "AvgPool1D",
    "AvgPool2D",
    "Flatten",
    "GRU",
    "LSTM",
    "RNN",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "sinusoidal_positions",
]

from dense_armor.utility.neural.batching import minibatches
from dense_armor.utility.neural.schedules import (
    ConstantLR,
    CosineWarmRestarts,
    StepLR,
)
from dense_armor.utility.neural.trainer import Trainer

__all__ = list(__all__) + [
    "minibatches",
    "ConstantLR",
    "StepLR",
    "CosineWarmRestarts",
    "Trainer",
]
